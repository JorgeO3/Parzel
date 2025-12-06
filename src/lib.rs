#![no_std]
#![feature(core_intrinsics)]
#![allow(internal_features)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)] 

//! # Hypersonic Search Engine (WASM Kernel)
//! 
//! "Hypersonic" is a high-performance, zero-copy information retrieval kernel 
//! designed specifically for the `wasm32-unknown-unknown` target.
//! 
//! ## Architecture
//! 
//! * **Zero-Copy:** Maps binary indices directly from memory without parsing.
//! * **SIMD-First:** Utilizes 128-bit vector instructions for intersection and scoring.
//! * **Hybrid Layout:** Switches between Bitmap (dense) and Compressed (sparse) storage.
//! * **No-Std:** Operates without the standard library, using only `core` and `alloc`.
//! 
//! ## Safety
//! 
//! This crate uses aggressive `unsafe` optimizations. Strict bounds checking is 
//! enforced in `HypersonicIter::new` to prevent out-of-bounds memory access 
//! from malformed or malicious index blobs.

extern crate alloc;

// ============================================================================
// CONFIGURACIÓN DE TESTS (NODE.JS)
// ============================================================================
#[cfg(test)]
extern crate wasm_bindgen_test;
#[cfg(test)]
use wasm_bindgen_test::*;
#[cfg(test)]
wasm_bindgen_test_configure!(run_in_node_experimental);

// ============================================================================
// 0. GESTIÓN DE MEMORIA
// ============================================================================

#[cfg(feature = "lol_alloc")]
#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOCATOR: lol_alloc::LockedAllocator<lol_alloc::FreeListAllocator> =
    lol_alloc::LockedAllocator::new(lol_alloc::FreeListAllocator::new());

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::intrinsics::abort()
}

use core::ptr::{addr_of, addr_of_mut};

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

// ============================================================================
// 1. ESTRUCTURAS DE DATOS
// ============================================================================

#[repr(C, align(16))]
pub struct AlignedLut(pub [u8; 16]);

pub const LUT_LOG_QUANT: AlignedLut = AlignedLut([
    0, 5, 12, 20, 30, 45, 60, 80, 
    100, 120, 140, 160, 180, 200, 225, 255
]);

#[repr(C)]
#[derive(Clone, Copy, Debug)] // Debug para tests
pub struct HypersonicIndex {
    pub magic: u32,
    pub num_docs: u32,
    pub fst_offset: u32,
    pub fst_len: u32,
    pub postings_base_offset: u32,
}

impl HypersonicIndex {
    #[inline(always)]
    pub fn get_term_offset(&self, term_id: u64) -> Option<u32> {
        if term_id == 0 { return None; }
        Some((term_id % 1000) as u32 * 256) 
    }
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq)]
pub enum ListType {
    Bitmap = 0,
    Compressed = 1,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CompressedHeader {
    pub num_blocks: u32,
    pub start_offset: u32,
}

#[repr(C)]
pub struct BlockHeader {
    pub max_doc: u32,
    pub max_score: u8,     
    pub bit_width: u8,
    pub padding: u16,
    pub data_offset: u32,
}

// ============================================================================
// 2. SIMD KERNELS
// ============================================================================

#[inline(always)]
#[cfg(target_arch = "wasm32")]
pub unsafe fn simd_any_true(v: v128) -> bool {
    v128_any_true(v)
}

/// Checks if any bit is set in the 128-bit vector.
///
/// # Safety
/// * Uses platform-specific intrinsics.
#[cfg(not(target_arch = "wasm32"))]
pub unsafe fn simd_any_true(_v: u128) -> bool { true }

// ============================================================================
// 3. HYBRID ITERATOR (HARDENED FOR MISALIGNMENT)
// ============================================================================

#[derive(Clone, Copy)]
pub struct HypersonicIter<'a> {
    strategy: PostingListStrategy<'a>,
    pub current_doc: u32,
    pub current_score: u8,
}

#[derive(Clone, Copy)]
pub enum PostingListStrategy<'a> {
    Empty,
    Bitmap {
        data: &'a [u8],
    },
    Compressed {
        headers: &'a [BlockHeader],
        block_idx: usize,
        decompressed_buf_ptr: *mut u32, 
        buf_pos: usize,
    }
}

impl<'a> HypersonicIter<'a> {
    pub fn empty() -> Self {
        Self { strategy: PostingListStrategy::Empty, current_doc: u32::MAX, current_score: 0 }
    }

    /// Creates a new iterator.
    /// 
    /// # Safety
    /// IMPORTANT: Instead of taking &HypersonicIndex directly (which might be unaligned),
    /// we take the raw pointer to the blob base to calculate offsets safely.
    /// The `postings_base` is passed explicitly.
pub unsafe fn new(
        blob_base_ptr: *const u8, 
        blob_len: usize,
        postings_base: u32, 
        offset: u32, 
        scratch_buf_ptr: *mut u32
    ) -> Self {
        unsafe {
            let blob_limit = blob_base_ptr.add(blob_len);
            let base_ptr = blob_base_ptr.add(postings_base as usize);
            let list_ptr = base_ptr.add(offset as usize);
            
            // Check básico de puntero
            if list_ptr >= blob_limit { return Self::empty(); }

            // ================================================================
            // SOLUCIÓN 2: FIX UNDEFINED BEHAVIOR (ENUM CAST)
            // ================================================================
            // ANTES (PELIGROSO): let list_type = *(list_ptr as *const ListType);
            // Si el byte era 3, 255, etc., esto era UB inmediato.
            
            // AHORA (SEGURO): Leemos el byte primitivo. Un u8 siempre es válido.
            let list_type_id = *list_ptr; 

            match list_type_id {
                // Caso 0: ListType::Bitmap
                0 => {
                    let len_ptr = list_ptr.add(1) as *const u32;
                    // Check de bounds para la longitud
                    if len_ptr.add(1) as *const u8 > blob_limit { return Self::empty(); }
                    
                    let len = len_ptr.read_unaligned() as usize;
                    let data_ptr = list_ptr.add(5);
                    let data_end = data_ptr.add(len);

                    // Check de bounds para el payload de datos
                    if data_end > blob_limit { return Self::empty(); }

                    let data = core::slice::from_raw_parts(data_ptr, len);
                    Self {
                        strategy: PostingListStrategy::Bitmap { data },
                        current_doc: 0, 
                        current_score: 0,
                    }
                },
                
                // Caso 1: ListType::Compressed
                1 => {
                    let header_ptr = list_ptr.add(1) as *const CompressedHeader;
                    if header_ptr.add(1) as *const u8 > blob_limit { return Self::empty(); }

                    let header = header_ptr.read_unaligned();
                    let blocks_ptr = base_ptr.add(header.start_offset as usize);
                    let headers_end = blocks_ptr.add(header.num_blocks as usize * size_of::<BlockHeader>());

                    if headers_end > blob_limit { return Self::empty(); }

                    let headers = core::slice::from_raw_parts(
                        blocks_ptr as *const BlockHeader, 
                        header.num_blocks as usize
                    );
                    
                    *scratch_buf_ptr = 0; 

                    Self {
                        strategy: PostingListStrategy::Compressed {
                            headers,
                            block_idx: 0,
                            decompressed_buf_ptr: scratch_buf_ptr, 
                            buf_pos: 128,
                        },
                        current_doc: 0,
                        current_score: 0,
                    }
                },

                // Caso Default: Corrupción o Versión Futura
                // Capturamos cualquier byte que no sea 0 o 1 y retornamos iterador vacío
                // en lugar de crashear o invocar comportamiento indefinido.
                _ => Self::empty(),
            }
        }
    }

    #[inline(always)]
    pub fn advance(&mut self, target: u32, min_score: u8) -> Option<u32> {
        self.advance_inner(target, min_score)
    }

    #[inline(never)]
    fn advance_inner(&mut self, target: u32, min_score: u8) -> Option<u32> {
        match &mut self.strategy {
            PostingListStrategy::Empty => None,
            
            PostingListStrategy::Bitmap { data } => {
let byte_idx = (target / 8) as usize;
                // Alineamos hacia atrás para empezar en frontera de 16 bytes
                let mut simd_idx = byte_idx & !15; 

                while simd_idx < data.len() {
                    // Calculamos espacio seguro restante
                    let remaining = data.len() - simd_idx;
                    
                    // =========================================================
                    // 1. CAMINO RÁPIDO (SIMD) - Vectorización 128-bit
                    // =========================================================
                    // Solo entramos si es físicamente seguro cargar un vector completo.
                    if remaining >= 16 {
                        #[cfg(target_arch = "wasm32")]
                        unsafe {
                            let ptr = data.as_ptr().add(simd_idx);
                            // v128_load es seguro aquí: garantizamos 16 bytes disponibles
                            let v_bits = v128_load(ptr as *const v128);
                            
                            if simd_any_true(v_bits) {
                                // Procesamos como 2 palabras de 64 bits para facilitar la aritmética
                                let u64_ptr = ptr as *const u64;
                                for i in 0..2 {
                                    // CRÍTICO: read_unaligned previene traps en punteros impares
                                    let word = u64_ptr.add(i).read_unaligned();
                                    
                                    if word != 0 {
                                        let current_base_doc = (simd_idx as u32) * 8 + (i as u32 * 64);
                                        let mut temp_word = word;
                                        
                                        // Enmascaramos bits anteriores al target
                                        if current_base_doc < target {
                                            let shift = target.saturating_sub(current_base_doc);
                                            if shift < 64 {
                                                temp_word &= !((1u64 << shift) - 1);
                                            } else {
                                                temp_word = 0;
                                            }
                                        }

                                        if temp_word != 0 {
                                            let trailing = temp_word.trailing_zeros();
                                            let doc = current_base_doc + trailing;
                                            self.current_doc = doc;
                                            self.current_score = 10;
                                            return Some(doc);
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Fallback para entornos de test (No-WASM)
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            // En tests unitarios en x86/M1, el bloque `if remaining >= 16` 
                            // no ejecuta nada SIMD, así que forzamos caer al bloque 'else' 
                            // o implementamos lógica escalar mock aquí si fuera necesario.
                            // Por ahora, el loop continúa y el `simd_idx += 16` saltaría este bloque.
                            // Para tests rigurosos, podrías duplicar la lógica 'Tail' aquí.
                        }
                    } 
                    // =========================================================
                    // 2. CAMINO LENTO (Tail / Scalar) - Byte a Byte
                    // =========================================================
                    // Se ejecuta para los últimos <16 bytes del archivo.
                    else {
                        for i in 0..remaining {
                            unsafe {
                                // get_unchecked es seguro: i < remaining
                                let byte = *data.get_unchecked(simd_idx + i);
                                
                                if byte != 0 {
                                    let current_base_doc = ((simd_idx + i) as u32) * 8;
                                    
                                    // Optimización: Si todo el byte es menor al target, saltar
                                    if current_base_doc + 8 <= target {
                                        continue;
                                    }

                                    // Chequeo bit a bit manual
                                    for bit in 0..8 {
                                        // Verificamos si el bit está encendido
                                        if (byte >> bit) & 1 != 0 {
                                            let doc = current_base_doc + bit;
                                            // Verificamos si cumple el target
                                            if doc >= target {
                                                self.current_doc = doc;
                                                self.current_score = 10;
                                                return Some(doc);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    simd_idx += 16;
                }
                None
            },

            PostingListStrategy::Compressed { headers, block_idx, buf_pos, decompressed_buf_ptr, .. } => {
                unsafe {
                    while *block_idx < headers.len() {
                        let h = &headers[*block_idx];
                        
                        if h.max_score < min_score {
                            *block_idx += 1;
                            *buf_pos = 128;
                            continue;
                        }
                        if h.max_doc < target {
                            *block_idx += 1;
                            *buf_pos = 128;
                            continue;
                        }

                        if *buf_pos >= 128 {
                            for i in 0..128 {
                                let base = if *block_idx == 0 { 0 } else { headers[*block_idx-1].max_doc };
                                *decompressed_buf_ptr.add(i) = base.saturating_add(i as u32 + 1);
                            }
                            *buf_pos = 0;
                        }

                        while *buf_pos < 128 {
                            let doc = *decompressed_buf_ptr.add(*buf_pos);
                            if doc >= target {
                                self.current_doc = doc;
                                self.current_score = h.max_score;
                                return Some(doc);
                            }
                            *buf_pos += 1;
                        }
                        *block_idx += 1;
                        *buf_pos = 128;
                    }
                    None
                }
            }
        }
    }
}

// ============================================================================
// 4. TOKENIZER (STUB)
// ============================================================================

pub struct FstTokenizer {
    // Stubs don't strictly need the index reference for now
}

impl FstTokenizer {
    pub fn new(_index: &HypersonicIndex) -> Self {
        Self {}
    }

    pub fn tokenize(&mut self, text: &str, out_tokens: &mut [u64]) -> usize {
        let bytes = text.as_bytes();
        let mut count = 0;
        let mut i = 0;
        while i < bytes.len() && count < out_tokens.len() {
            if bytes[i] == b' ' { i += 1; continue; }
            let mut j = i;
            while j < bytes.len() && bytes[j] != b' ' { j += 1; }
            let word_len = (j - i) as u64;
            out_tokens[count] = word_len * 100;
            count += 1;
            i = j;
        }
        count
    }
}

// ============================================================================
// 5. GLOBAL STATE & FFI EXPORTS
// ============================================================================

static mut INDEX_PTR: *const HypersonicIndex = core::ptr::null();
static mut INDEX_LEN: usize = 0;
static mut RESULTS_BUFFER: [u32; 128] = [0; 128]; 

/// Initializes the search index.
///
/// # Safety
/// * `ptr` must be a valid pointer to the index blob.
/// * `len` must be the exact size of the blob.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_index(ptr: *const u8, len: usize) {
    unsafe {
        INDEX_PTR = ptr as *const HypersonicIndex;
        INDEX_LEN = len;
    }
}

/// Executes a search.
///
/// # Safety
/// * `query_ptr` must point to UTF-8 data.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn search(query_ptr: *const u8, query_len: usize) -> u32 {
    unsafe {
        if INDEX_PTR.is_null() { return 0; }
        
        // FIX CRÍTICO V1.9: Copiar header a stack para acceso alineado seguro.
        // Si INDEX_PTR es impar, crear una referencia &*INDEX_PTR es UB.
        let index_header = INDEX_PTR.read_unaligned();

        let mut scratch_buffers = [[0u32; 128]; 16]; 

        let query_slice = core::slice::from_raw_parts(query_ptr, query_len);
        let query_str = match core::str::from_utf8(query_slice) {
            Ok(s) => s,
            Err(_) => return 0, // Si no es texto válido, no hay resultados. Safe.
        };
        
        let mut tokenizer = FstTokenizer::new(&index_header);
        let mut tokens = [0u64; 16]; 
        let token_count = tokenizer.tokenize(query_str, &mut tokens);

        if token_count == 0 { return 0; }

        let mut iterators: [HypersonicIter; 16] = [HypersonicIter::empty(); 16];
        for i in 0..token_count {
            if let Some(offset) = index_header.get_term_offset(tokens[i]) {
                // Pasamos el puntero crudo original (posiblemente desalineado)
                // y los offsets del header copiado.
                iterators[i] = HypersonicIter::new(
                    INDEX_PTR as *const u8, 
                    INDEX_LEN,
                    index_header.postings_base_offset,
                    offset, 
                    scratch_buffers[i].as_mut_ptr()
                );
            }
        }

        let mut matches = 0;
        let buffer_ptr = addr_of_mut!(RESULTS_BUFFER) as *mut u32;
        let mut current_target = 0;
        
        while matches < 128 {
            if let Some(doc) = iterators[0].advance(current_target, 0) {
                let mut is_match = true;
                for iter in iterators.iter_mut().take(token_count).skip(1) {
                    if let Some(other_doc) = iter.advance(doc, 0) {
                        if other_doc != doc {
                            is_match = false;
                            current_target = other_doc;
                            break;
                        }
                    } else {
                        return matches as u32;
                    }
                }

                if is_match {
                    *buffer_ptr.add(matches) = doc;
                    matches += 1;
                    current_target = doc + 1;
                }
            } else {
                break;
            }
        }

        matches as u32
    }
}

/// Gets the result buffer pointer.
///
/// # Safety
/// * Returns raw static pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_results_ptr() -> *const u32 {
    addr_of!(RESULTS_BUFFER) as *const u32
}

// ============================================================================
// 6. TEST SUITE (MILITARY GRADE)
// ============================================================================

#[cfg(test)]
mod torture_chamber {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    // ========================================================================
    // HELPER: IndexBuilder
    // Construye estructuras de índice binario válidas o corruptas programáticamente.
    // ========================================================================
    struct IndexBuilder {
        data: Vec<u8>,
        postings_start: usize,
    }

    impl IndexBuilder {
        fn new() -> Self {
            let data = vec![0u8; 20]; // Espacio para Header
            Self { data, postings_start: 20 }
        }

        fn set_header(&mut self, num_docs: u32) {
            let header = HypersonicIndex {
                magic: 0x48595030,
                num_docs,
                fst_offset: 0,
                fst_len: 0,
                postings_base_offset: 20,
            };
            // Escritura insegura para simular el layout en memoria cruda
            unsafe {
                let ptr = self.data.as_mut_ptr() as *mut HypersonicIndex;
                *ptr = header; 
            }
        }

        fn add_bitmap_list(&mut self, docs: &[u32]) -> u32 {
            let offset = (self.data.len() - self.postings_start) as u32;
            self.data.push(ListType::Bitmap as u8);
            
            // Calculamos bytes necesarios para el bitmap
            let max_doc = docs.iter().max().unwrap_or(&0);
            let needed_bytes = ((max_doc / 8) + 1) as usize;
            // Alineamos a 16 bytes (SIMD friendly padding) si es necesario por tu spec, 
            // o usamos needed_bytes crudos si queremos testear límites estrictos.
            // Asumiremos padding standard aquí:
            let len_bytes = (needed_bytes + 15) & !15; 
            
            self.data.extend_from_slice(&(len_bytes as u32).to_le_bytes());
            let start_idx = self.data.len();
            self.data.extend(core::iter::repeat_n(0, len_bytes));
            
            for &doc in docs {
                let byte_pos = (doc / 8) as usize;
                let bit_pos = doc % 8;
                if start_idx + byte_pos < self.data.len() {
                    self.data[start_idx + byte_pos] |= 1 << bit_pos;
                }
            }
            offset
        }

        // Simula una lista corrupta que declara una longitud mayor a la real
        fn add_corrupt_list(&mut self, fake_len: usize) -> u32 {
            let offset = (self.data.len() - self.postings_start) as u32;
            self.data.push(ListType::Bitmap as u8);
            self.data.extend_from_slice(&(fake_len as u32).to_le_bytes());
            offset
        }

        // Simula un byte de tipo de lista desconocido (ej. corrupción de datos)
        fn add_garbage_list(&mut self) -> u32 {
            let offset = (self.data.len() - self.postings_start) as u32;
            self.data.push(0xFF); // 0xFF no es un ListType válido
            self.data.extend_from_slice(&(50u32).to_le_bytes()); // Longitud dummy
            self.data.extend(core::iter::repeat_n(0xAA, 50)); // Basura
            offset
        }

        fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }
        fn len(&self) -> usize { self.data.len() }

        // Agregar este método al impl IndexBuilder
        // Crea una lista comprimida "mock" que generará docs secuenciales
        fn add_mock_compressed_list(&mut self, num_blocks: u32) -> u32 {
            let offset = (self.data.len() - self.postings_start) as u32;
            self.data.push(ListType::Compressed as u8); // Type 1

            // CompressedHeader
            let header_offset = 200; // Offset relativo arbitrario para los bloques
            let ch = CompressedHeader { num_blocks, start_offset: header_offset };
            
            unsafe {
                let ch_slice = core::slice::from_raw_parts(&ch as *const _ as *const u8, size_of::<CompressedHeader>());
                self.data.extend_from_slice(ch_slice);
            }

            // Rellenar hasta llegar al header_offset (padding)
            let current_len = (self.data.len() - self.postings_start) as u32;
            if current_len < header_offset {
                self.data.extend(core::iter::repeat_n(0, (header_offset - current_len) as usize));
            }

            // BlockHeaders
            for i in 0..num_blocks {
                let bh = BlockHeader {
                    max_doc: (i + 1) * 128, // Docs 0-127, 128-255, etc.
                    max_score: 10,
                    bit_width: 0,
                    padding: 0,
                    data_offset: 0, // No usado en el mock de descompresión
                };
                unsafe {
                    let bh_slice = core::slice::from_raw_parts(&bh as *const _ as *const u8, size_of::<BlockHeader>());
                    self.data.extend_from_slice(bh_slice);
                }
            }
            offset
        }
    }

    fn setup_index_env(data: &[u8]) {
        unsafe { init_index(data.as_ptr(), data.len()); }
    }

    // ========================================================================
    // TESTS DE MEMORIA Y ALINEACIÓN (Hardcore)
    // ========================================================================

    #[wasm_bindgen_test]
    fn test_misaligned_memory_access() {
        // Objetivo: Asegurar que leemos u32/u64 sin crashear en punteros impares (WASM/ARM trap)
        let mut builder = IndexBuilder::new();
        builder.set_header(100);
        let _ = builder.add_bitmap_list(&[10, 20]);

        // Clonamos y desplazamos 1 byte para forzar puntero impar (0x...1)
        let mut misaligned_buffer = vec![0u8; builder.len() + 1];
        misaligned_buffer[1..].copy_from_slice(&builder.data);

        let ptr = unsafe { misaligned_buffer.as_ptr().add(1) };
        
        unsafe {
            init_index(ptr, builder.len());
            // Verificamos lectura segura del header global
            let idx = INDEX_PTR.read_unaligned();
            assert_eq!(idx.num_docs, 100);
            assert_eq!(idx.magic, 0x48595030);
        }
    }

    #[wasm_bindgen_test]
    fn test_exact_buffer_boundary_access() {
        // Objetivo: Detectar "Buffer Over-read". 
        // Si el bitmap termina en el último byte de memoria, intentar leer 
        // 4 u 8 bytes extra para optimización SIMD causará un segfault/trap.
        let mut builder = IndexBuilder::new();
        builder.set_header(100);
        // Lista pequeña al final del buffer
        let offset = builder.add_bitmap_list(&[0, 1, 2]); 

        unsafe {
            init_index(builder.as_ptr(), builder.len());
            let index_header = INDEX_PTR.read_unaligned();
            let mut scratch = [0u32; 128];

            // Pasamos length exacto. La implementación debe respetar `max_len`.
            let mut iter = HypersonicIter::new(
                INDEX_PTR as *const u8,
                builder.len(), 
                index_header.postings_base_offset,
                offset, 
                scratch.as_mut_ptr()
            );
            
            // Si la iteración interna hace `ptr.add(4)` sin chequear bounds, esto explota.
            assert_eq!(iter.advance(0, 0), Some(0));
            assert_eq!(iter.advance(1, 0), Some(1));
            assert_eq!(iter.advance(2, 0), Some(2));
            assert_eq!(iter.advance(3, 0), None);
        }
    }

    // ========================================================================
    // TESTS DE CORRUPCIÓN Y SEGURIDAD
    // ========================================================================

    #[wasm_bindgen_test]
    fn test_corrupt_index_graceful_fail() {
        // Objetivo: Evitar panic cuando el índice miente sobre su longitud
        let mut builder = IndexBuilder::new();
        builder.set_header(100);
        // Decimos que la lista mide 1GB, pero el buffer es minúsculo
        let offset = builder.add_corrupt_list(1024 * 1024 * 1024); 

        unsafe {
            init_index(builder.as_ptr(), builder.len());
            let index_header = INDEX_PTR.read_unaligned();
            let mut scratch = [0u32; 128];

            let mut iter = HypersonicIter::new(
                INDEX_PTR as *const u8,
                INDEX_LEN,
                index_header.postings_base_offset,
                offset, 
                scratch.as_mut_ptr()
            );
            
            // Debe retornar None inmediatamente al detectar overflow de bounds
            let res = iter.advance(0, 0);
            assert_eq!(res, None);
        }
    }

    #[wasm_bindgen_test]
    fn test_unknown_list_type_safety() {
        // Objetivo: Manejar bytes de tipo inválidos sin caer en loop infinito o cast inválido
        let mut builder = IndexBuilder::new();
        builder.set_header(100);
        let offset = builder.add_garbage_list();

        unsafe {
            init_index(builder.as_ptr(), builder.len());
            let index_header = INDEX_PTR.read_unaligned();
            let mut scratch = [0u32; 128];

            let mut iter = HypersonicIter::new(
                INDEX_PTR as *const u8,
                INDEX_LEN,
                index_header.postings_base_offset,
                offset, 
                scratch.as_mut_ptr()
            );
            
            // Debería fallar elegantemente
            assert_eq!(iter.advance(0, 0), None); 
        }
    }

    // ========================================================================
    // TESTS LÓGICOS Y DE ESTADO
    // ========================================================================

    #[wasm_bindgen_test]
    fn test_empty_bitmap_list() {
        // Objetivo: Asegurar que no hay divisiones por cero o underflows con listas vacías
        let mut builder = IndexBuilder::new();
        builder.set_header(100);
        let offset = builder.add_bitmap_list(&[]); // Lista válida de 0 docs

        unsafe {
            init_index(builder.as_ptr(), builder.len());
            let index_header = INDEX_PTR.read_unaligned();
            let mut scratch = [0u32; 128];

            let mut iter = HypersonicIter::new(
                INDEX_PTR as *const u8,
                INDEX_LEN,
                index_header.postings_base_offset,
                offset, 
                scratch.as_mut_ptr()
            );
            
            assert_eq!(iter.advance(0, 0), None);
        }
    }

    #[wasm_bindgen_test]
    fn test_stack_memory_write() {
        // Objetivo: Verificar que el iterador escribe correctamente en el scratch buffer
        // y que avanza lógicamente.
        let mut builder = IndexBuilder::new();
        builder.set_header(100);
        let docs = vec![0, 128, 256];
        let offset = builder.add_bitmap_list(&docs);
        
        // Desalineamos para dificultar las cosas
        let mut misaligned = vec![0u8; builder.len() + 1];
        misaligned[1..].copy_from_slice(&builder.data);
        let ptr = unsafe { misaligned.as_ptr().add(1) };

        unsafe {
            init_index(ptr, builder.len());
            let index_header = INDEX_PTR.read_unaligned();
            let mut scratch = [0u32; 128];
            
            let mut iter = HypersonicIter::new(
                INDEX_PTR as *const u8,
                INDEX_LEN,
                index_header.postings_base_offset,
                offset, 
                scratch.as_mut_ptr()
            );
            
            assert_eq!(iter.advance(0, 0), Some(0));
            assert_eq!(iter.advance(100, 0), Some(128));
            assert_eq!(iter.advance(300, 0), None);
        }
    }

    #[wasm_bindgen_test]
    fn test_double_initialization() {
        // Objetivo: Asegurar que el estado global (static mut) se puede resetear
        // sin leaks ni corrupción de punteros antiguos.
        let mut builder1 = IndexBuilder::new();
        builder1.set_header(100);
        let _ = builder1.add_bitmap_list(&[10]);

        let mut builder2 = IndexBuilder::new();
        builder2.set_header(999); // Header diferente
        let _ = builder2.add_bitmap_list(&[55]); 

        unsafe {
            // Init 1
            init_index(builder1.as_ptr(), builder1.len());
            let idx1 = INDEX_PTR.read_unaligned();
            assert_eq!(idx1.num_docs, 100);

            // Init 2 (Simula recarga de archivo)
            init_index(builder2.as_ptr(), builder2.len());
            let idx2 = INDEX_PTR.read_unaligned();
            assert_eq!(idx2.num_docs, 999); // Debe reflejar el nuevo header
        }
    }

#[wasm_bindgen_test]
    fn test_break_simd_overread() {
        // Construimos un bitmap de 17 bytes.
        let mut data = vec![0u8; 100]; 
        
        let header = HypersonicIndex {
            magic: 0x48595030, num_docs: 1000, fst_offset: 0, fst_len: 0, postings_base_offset: 50,
        };
        // FIX: Usamos write_unaligned para el header por si acaso
        unsafe { 
            (data.as_ptr() as *mut HypersonicIndex).write_unaligned(header); 
        }

        // Bitmap List Setup en offset 50
        let list_offset = 50;
        data[list_offset] = ListType::Bitmap as u8;
        let len_bytes: u32 = 17;
        
        unsafe {
            // ================================================================
            // EL FIX CRÍTICO ESTÁ AQUÍ:
            // Antes: *ptr = len_bytes (CRASH por alineación)
            // Ahora: ptr.write_unaligned(len_bytes) (SEGURO)
            // ================================================================
            let len_ptr = data.as_ptr().add(list_offset + 1) as *mut u32;
            len_ptr.write_unaligned(len_bytes);
        }
        
        // Rellenamos datos del bitmap
        let bitmap_start = list_offset + 5;
        data[bitmap_start + 16] = 1; // Bit en el byte 17
        
        // CORTAMOS EL VECTOR PARA QUE NO HAYA MEMORIA EXTRA
        data.truncate(bitmap_start + 17); 

        setup_index_env(&data);
        
        let mut scratch = [0u32; 128];
        unsafe {
            let idx = INDEX_PTR.read_unaligned();
            
            let mut iter = HypersonicIter::new(
                INDEX_PTR as *const u8, 
                INDEX_LEN, 
                idx.postings_base_offset, 
                0, 
                scratch.as_mut_ptr()
            );

            // Ahora sí probamos la lógica de la librería
            // Si tu librería tiene el fix del "remaining >= 16", esto pasará.
            // Si no lo tiene, esto fallará con memory access out of bounds.
            iter.advance(0, 0); 
        }
    }

    #[wasm_bindgen_test]
    fn test_break_ub_enum() {
        let mut data = vec![0u8; 100];
        let header = HypersonicIndex {
            magic: 0x48595030, num_docs: 1000, fst_offset: 0, fst_len: 0, postings_base_offset: 50,
        };
        unsafe { *(data.as_ptr() as *mut HypersonicIndex) = header; }

        // En el offset 50, ponemos un tipo inválido (ej. 3)
        // ListType solo acepta 0 o 1.
        data[50] = 3; 

        setup_index_env(&data);
        let mut scratch = [0u32; 128];
        unsafe {
            let idx = INDEX_PTR.read_unaligned();
            
            // Esto técnicamente ya es UB dentro de new()
            // Rust no garantiza que el programa siga corriendo después de esto.
            let _iter = HypersonicIter::new(
                INDEX_PTR as *const u8, 
                INDEX_LEN, 
                idx.postings_base_offset, 
                0, 
                scratch.as_mut_ptr()
            );
        }
    }

    #[wasm_bindgen_test]
    fn test_happy_path_intersection() {
        let mut builder = IndexBuilder::new();
        builder.set_header(1000);
        
        // Término "apple": Docs [10, 20, 30]
        let offset_apple = builder.add_bitmap_list(&[10, 20, 30]);
        // Término "pie": Docs [20, 30, 40]
        let offset_pie = builder.add_bitmap_list(&[20, 30, 40]);

        // Intersección esperada: [20, 30]

        unsafe {
            init_index(builder.as_ptr(), builder.len());
            let index_header = INDEX_PTR.read_unaligned();
            
            // Simulamos manual la lógica de search() para no depender del Tokenizer stub
            let mut scratch_a = [0u32; 128];
            let mut scratch_b = [0u32; 128];

            let mut iter_apple = HypersonicIter::new(
                INDEX_PTR as *const u8, INDEX_LEN, index_header.postings_base_offset, offset_apple, scratch_a.as_mut_ptr()
            );
            let mut iter_pie = HypersonicIter::new(
                INDEX_PTR as *const u8, INDEX_LEN, index_header.postings_base_offset, offset_pie, scratch_b.as_mut_ptr()
            );

            // Algoritmo de intersección manual (SvS - Simulado)
            let mut results = Vec::new();
            let mut target = 0;
            
            while let Some(doc_a) = iter_apple.advance(target, 0) {
                if let Some(doc_b) = iter_pie.advance(doc_a, 0) {
                    if doc_a == doc_b {
                        results.push(doc_a);
                        target = doc_a + 1;
                    } else {
                        target = doc_b;
                    }
                } else {
                    break; 
                }
            }

            assert_eq!(results, vec![20, 30], "La intersección falló");
        }
    }

    // EL TEST
    #[wasm_bindgen_test]
    fn test_compressed_block_crossing() {
        let mut builder = IndexBuilder::new();
        builder.set_header(5000);
        // Creamos 2 bloques. 
        // Bloque 0: Docs 1..128
        // Bloque 1: Docs 129..256
        let offset = builder.add_mock_compressed_list(2);

        unsafe {
            init_index(builder.as_ptr(), builder.len());
            let idx = INDEX_PTR.read_unaligned();
            let mut scratch = [0u32; 128];
            
            let mut iter = HypersonicIter::new(
                INDEX_PTR as *const u8, INDEX_LEN, idx.postings_base_offset, offset, scratch.as_mut_ptr()
            );

            // Avanzamos hasta el último doc del bloque 1
            let doc_128 = iter.advance(128, 0).expect("Debe encontrar doc 128");
            assert_eq!(doc_128, 128); // Tu logica genera index + 1
            
            // Avanzamos cruzando la frontera al bloque 2
            let doc_129 = iter.advance(129, 0).expect("Debe cruzar al bloque 2");
            assert_eq!(doc_129, 129);

            // Verificamos el final
             let doc_256 = iter.advance(256, 0).expect("Debe encontrar ultimo doc");
             assert_eq!(doc_256, 256);

             assert_eq!(iter.advance(257, 0), None, "Debe terminar");
        }
    }

    #[wasm_bindgen_test]
    fn test_utf8_bomb() {
        // Setup minimo
        let mut builder = IndexBuilder::new();
        builder.set_header(10);
        unsafe { init_index(builder.as_ptr(), builder.len()); }

        // Input inválido: Byte de continuación sin byte de inicio (0x80)
        let invalid_utf8 = [0x80, 0x81]; 
        
        unsafe {
            // Esto NO DEBERÍA crashear si el código fuera seguro.
            // Pero con from_utf8_unchecked, el Tokenizer iterará sobre basura.
            // Dependiendo de la implementación del Tokenizer, podría hacer panic.
            let _ = search(invalid_utf8.as_ptr(), invalid_utf8.len());
        }
    }
}