#![allow(unused)]

use alloc::string::String;
use core::num::TryFromIntError;

/// El tipo Result global para tu librería.
/// Ahorra escribir `Result<T, Error>` en todas partes.
pub type Result<T> = core::result::Result<T, Error>;

/// El error principal de Parzel.
///
/// Derive `Debug` y `thiserror::Error` para manejo automático.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Errores genéricos o mensajes de pánico controlados (como 'anyhow' pero tipado).
    #[error("Engine error: {0}")]
    Generic(String),

    /// Errores de formato en el índice binario (Magic bytes, versiones, etc).
    #[error("Corrupted index data: {0}")]
    CorruptedData(&'static str),

    /// El dato es demasiado grande para el formato o la arquitectura (u32 overflow).
    /// #[from] convierte automáticamente el error de `try_from` a esta variante.
    #[error("Value overflow: {0}")]
    Overflow(#[from] TryFromIntError),

    /// Errores de UTF-8 al parsear queries.
    #[error("Invalid UTF-8 sequence")]
    Utf8(#[from] core::str::Utf8Error),
    // Si usas serde_yml o json en el futuro, actívalos condicionalmente:
    // #[cfg(feature = "serialization")]
    // #[error("Serialization error: {0}")]
    // Serialization(String),
}

// Helper para convertir &str a Error::Generic fácilmente
impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Generic(alloc::string::String::from(s))
    }
}
