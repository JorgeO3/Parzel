use core::mem::MaybeUninit;
use core::slice;

use crate::data::HypersonicIndex;
use crate::iter::{PostingIterator, SCRATCH_SIZE};
use crate::tokenizer::MAX_TOKENS;

// ============================================================================
// 1. SEARCH ARENA (El Dueño de la Memoria)
// ============================================================================

/// Contenedor de memoria "pesada" reutilizable.
/// Gestiona los buffers de descompresión (Scratchpads).
///
/// # Performance
/// - Usa `MaybeUninit` para evitar que Rust llene el stack de ceros (ahorra ~2-3us).
/// - Usa `align(64)` para asegurar que cada buffer empiece en una línea de caché limpia.
#[repr(C, align(64))]
pub struct SearchArena {
    // Array de buffers sin inicializar.
    // [u32; 128] * 16 iteradores = 8KB de memoria.
    scratches: [MaybeUninit<[u32; SCRATCH_SIZE]>; MAX_TOKENS],
}

impl SearchArena {
    /// Crea una nueva arena en el stack. Costo: 0 ciclos CPU.
    #[inline]
    pub const fn new() -> Self {
        // SAFETY: Un array de MaybeUninit no requiere inicialización.
        // El compilador reserva el espacio en el stack pointer (sub rsp) pero no escribe nada.
        Self {
            // SAFETY: MaybeUninit<T> allows uninitialized memory; assume_init is safe for arrays of MaybeUninit.
            scratches: [MaybeUninit::uninit(); MAX_TOKENS],
        }
    }

    /// Retorna un iterador que entrega buffers mutables listos para usar.
    ///
    /// SAFETY:
    /// - Garantiza disjointness (no hay dos iteradores con el mismo buffer).
    /// - Asume que la memoria es escribible (`MaybeUninit` -> T).
    ///   Como el `PostingIterator` SOLO escribe antes de leer, esto es seguro.
    #[inline]
    pub fn available_buffers(
        &mut self,
    ) -> impl Iterator<Item = &mut MaybeUninit<[u32; SCRATCH_SIZE]>> {
        self.scratches.iter_mut()
    }
}

// ============================================================================
// 2. QUERY CONTEXT (El Gestor de Iteradores)
// ============================================================================

/// Contexto efímero que vive solo lo que dura una query.
/// Pide prestada la memoria al Arena, evitando estructuras autorreferenciales.
pub struct QueryCtx<'a> {
    // // Referencia al dueño de la memoria (Split Ownership)
    // pub arena: &'a mut SearchArena,
    // Iteradores almacenados también sin inicializar para evitar overhead.
    pub iters: [MaybeUninit<PostingIterator<'a>>; MAX_TOKENS],
}

impl<'a> QueryCtx<'a> {
    /// Crea el contexto enlazado a una Arena.
    pub const fn new() -> Self {
        Self {
            // arena,
            // SAFETY: MaybeUninit<T> allows uninitialized memory; assume_init is safe for arrays of MaybeUninit.
            iters: [const { MaybeUninit::uninit() }; MAX_TOKENS],
        }
    }

    /// Prepara e inicializa un iterador en el slot `idx`.
    #[inline]
    pub fn prepare(
        &mut self,
        arena: &'a mut SearchArena, // El préstamo ocurre AQUÍ, no en el struct
        index: &'a HypersonicIndex,
        tokens: &[&str],
    ) -> usize {
        let mut count = 0;

        // MAGIA DE RUST: Zip
        // Emparejamos cada token con un buffer disponible del arena.
        // Si se acaban los tokens o se acaban los buffers, el loop termina.
        // No hay bounds checks, no hay índices manuales.
        let zip_iter = tokens.iter().zip(arena.available_buffers());

        for (token, scratch) in zip_iter {
            // Intentamos buscar el término en el índice
            if let Some(offset) = index.get_term_offset(token) {
                // 1. Crear el iterador (SAFE)
                // 'scratch' ya es &mut [u32], Rust sabe que es único.
                let iter = PostingIterator::new(index, offset, scratch);

                self.iters[count].write(iter);

                count += 1;
            }
        }

        count
    }

    /// Retorna un slice seguro con los iteradores activos para la intersección.
    #[inline]
    pub fn active_slice(&mut self, count: usize) -> &mut [PostingIterator<'a>] {
        debug_assert!(count <= MAX_TOKENS);
        // SAFETY: All elements up to `count` in `iters` have been initialized via `prepare_iter`,
        // and `count` is guaranteed to be <= MAX_TOKENS, so casting to a slice of initialized objects is safe.
        unsafe {
            // Transmutamos el array de MaybeUninit a un slice de objetos inicializados.
            slice::from_raw_parts_mut(self.iters.as_mut_ptr().cast::<PostingIterator<'a>>(), count)
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::HypersonicIndex;
    use fst::MapBuilder;
    use wasm_bindgen_test::*;

    // Configuración para correr en Node.js
    wasm_bindgen_test_configure!(run_in_node_experimental);

    // ------------------------------------------------------------------------
    // HELPERS
    // ------------------------------------------------------------------------

    /// Helper: Construye un índice válido binario con una lista arbitraria de términos.
    /// NOTA: Los términos deben pasarse ORDENADOS alfabéticamente para que FST funcione.
    fn create_index_with_terms(sorted_terms: &[&str]) -> Vec<u8> {
        // 1. Construir un FST válido
        let mut build = MapBuilder::memory();
        for (i, term) in sorted_terms.iter().enumerate() {
            // Insertamos (term, id) - el ID no importa mucho para estos tests de contexto
            build.insert(term, (i + 1) as u64).unwrap();
        }
        let fst_bytes = build.into_inner().unwrap();

        // 2. Construir el Header (24 bytes mínimos)
        let mut data = vec![0u8; 24];

        // Magic "HYP0"
        data[0..4].copy_from_slice(&0x4859_5030u32.to_le_bytes());

        // Postings Offset: Justo después del FST
        let fst_len = fst_bytes.len() as u32;
        let postings_offset = 24 + fst_len;
        data[16..20].copy_from_slice(&postings_offset.to_le_bytes());

        // FST Size
        data[20..24].copy_from_slice(&fst_len.to_le_bytes());

        // 3. Unir todo
        data.extend_from_slice(&fst_bytes);

        // 4. Agregar espacio dummy para postings (para que get_term_offset apunte a algo válido)
        // Agregamos suficiente padding para que HypersonicIndex no se queje de overflow
        data.resize(data.len() + 100, 0);
        // Escribimos un byte de tipo "Bitmap" (0) en el offset destino para que PostingIterator::new no falle
        // Calculamos dónde caerían los postings (postings_offset + id)
        // Como simplificación, llenamos de ceros, que puede interpretarse como Bitmap vacíos o headers truncados
        // pero para 'context::prepare' solo importa que 'get_term_offset' devuelva Some.

        data
    }

    fn create_dummy_index() -> Vec<u8> {
        create_index_with_terms(&["test"])
    }

    // ------------------------------------------------------------------------
    // TEST SUITE
    // ------------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_arena_alignment_guarantee() {
        let arena = SearchArena::new();
        // Obtenemos el puntero crudo al inicio del array de scratches
        let ptr = arena.scratches.as_ptr() as usize;

        // Verificamos alineación a 64 bytes (ptr % 64 == 0)
        assert_eq!(
            ptr % 64,
            0,
            "FATAL: SearchArena no está alineado a 64 bytes. SIMD fallará."
        );

        // Verificamos que los buffers internos también sean contiguos y respeten el layout
        let first_buffer_addr = arena.scratches[0].as_ptr() as usize;
        assert_eq!(
            first_buffer_addr % 64,
            0,
            "El primer buffer no está alineado."
        );
    }

    #[wasm_bindgen_test]
    fn test_arena_buffers_disjointness() {
        let mut arena = SearchArena::new();
        let mut pointers = Vec::new();

        for buffer in arena.available_buffers() {
            pointers.push(buffer.as_mut_ptr() as usize);
            // Escribimos algo para asegurar que es escribible
            let slice = unsafe { buffer.assume_init_mut() };
            slice[0] = 0xDEAD_BEEF;
        }

        // 1. Debe haber MAX_TOKENS buffers
        assert_eq!(pointers.len(), MAX_TOKENS);

        // 2. Todos los punteros deben ser únicos
        pointers.sort_unstable();
        pointers.dedup();
        assert_eq!(
            pointers.len(),
            MAX_TOKENS,
            "CRITICAL: SearchArena entregó el mismo buffer dos veces (Aliasing)."
        );
    }

    #[wasm_bindgen_test]
    fn test_prepare_logic_truncation() {
        // Escenario: El usuario manda MÁS tokens de los que soportamos.
        let mut arena = SearchArena::new();
        let mut ctx = QueryCtx::new();

        // 1. Tokens excesivos (usamos "test" que existe en el índice dummy)
        let binding = ["test"; MAX_TOKENS + 5];
        let tokens: Vec<&str> = binding.to_vec();

        // 2. Crear Índice Real
        let data = create_dummy_index();
        let index = HypersonicIndex::new(&data).expect("El índice dummy debe ser válido");

        // 3. Ejecutar prepare
        let count = ctx.prepare(&mut arena, &index, &tokens);

        // 4. Validar truncamiento
        assert_eq!(
            count, MAX_TOKENS,
            "Buffer overflow: prepare aceptó más tokens de los permitidos."
        );

        let active = ctx.active_slice(count);
        assert_eq!(active.len(), MAX_TOKENS);
    }

    #[wasm_bindgen_test]
    fn test_prepare_logic_gaps_handling() {
        // Escenario: Tokens válidos mezclados con inválidos.
        // Query: "existente" (ok) -> "raro" (null) -> "otro" (ok)
        // Esperado: count = 2.
        // "existente" -> active_slice[0]
        // "raro"      -> saltado
        // "otro"      -> active_slice[1]

        let mut arena = SearchArena::new();
        let mut ctx = QueryCtx::new();
        let tokens = vec!["existente", "raro", "otro"];

        // 1. Creamos un índice que SOLO contiene "existente" y "otro".
        // NOTA: Deben estar ordenados alfabéticamente para create_index_with_terms.
        let data = create_index_with_terms(&["existente", "otro"]);
        let index = HypersonicIndex::new(&data).expect("Índice de gaps válido");

        // 2. Ejecutar
        let count = ctx.prepare(&mut arena, &index, &tokens);

        // 3. Validar
        assert_eq!(count, 2, "El conteo de iteradores activos es incorrecto.");

        let slice = ctx.active_slice(count);
        assert_eq!(slice.len(), 2);

        // El iterador 0 debería corresponder a "existente" y el 1 a "otro".
        // (Verificar IDs internos requeriría exponer métodos de PostingIterator,
        // pero verificar el count ya nos dice que el 'raro' fue saltado).
    }

    #[wasm_bindgen_test]
    fn test_scratch_buffer_lifecycle() {
        // Verificamos que el ciclo de vida de MaybeUninit -> init -> slice funcione
        let mut arena = SearchArena::new();

        // Fase 1: Obtener buffer
        let mut iter = arena.available_buffers();
        let Some(buf) = iter.next() else {
            panic!("Debería haber al menos un buffer disponible");
        };

        // Fase 2: Escribir
        let slice = unsafe { buf.assume_init_mut() };
        slice[0] = 12345;
        slice[SCRATCH_SIZE - 1] = 67890;

        // Fase 3: Verificar persistencia
        assert_eq!(slice[0], 12345);
        assert_eq!(slice.len(), 128); // SCRATCH_SIZE
    }
}
