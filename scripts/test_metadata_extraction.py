#!/usr/bin/env python3
"""
Script de prueba para verificar la extracción de metadatos de nombres de archivos PDF
"""
import re

def extract_metadata_from_filename(filename: str) -> dict:
    """
    Extrae metadatos del nombre del archivo PDF.
    """
    metadata = {
        'manufacturer': None,
        'model': None,
        'version': None
    }
    
    # Mapeo de prefijos conocidos a fabricantes
    manufacturer_patterns = {
        'AX': 'Beckhoff',
        'EL': 'Beckhoff',
        'EK': 'Beckhoff',
        'CX': 'Beckhoff',
        'Lexium': 'Schneider Electric',
        'Altivar': 'Schneider Electric',
        'Modicon': 'Schneider Electric',
        'CC': 'Unknown',  # Para CC103
    }
    
    # Remover la extensión .pdf
    name_without_ext = filename.replace('.pdf', '').replace('.PDF', '')
    
    # Buscar patrones de modelo
    # Patrón 1: Modelo al inicio seguido de underscore (ej: AX5000_SystemManual)
    model_match = re.match(r'^([A-Z]+[0-9]+[A-Z]*)_', name_without_ext)
    if model_match:
        metadata['model'] = model_match.group(1)
    else:
        # Patrón 2: Modelo con letras y números juntos (ej: Lexium32M)
        model_match = re.match(r'^([A-Za-z]+[0-9]+[A-Za-z]*)', name_without_ext)
        if model_match:
            metadata['model'] = model_match.group(1)
        else:
            # Patrón 3: Buscar después de underscore (ej: 072152-101_CC103_Hardware)
            model_match = re.search(r'_([A-Z]+[0-9]+)_', name_without_ext)
            if model_match:
                metadata['model'] = model_match.group(1)
    
    # Determinar fabricante basado en el modelo encontrado
    if metadata['model']:
        for prefix, manufacturer in manufacturer_patterns.items():
            if metadata['model'].startswith(prefix):
                metadata['manufacturer'] = manufacturer
                break
        
        # Si no se encontró fabricante, usar "Unknown"
        if not metadata['manufacturer']:
            metadata['manufacturer'] = 'Unknown'
    
    # Buscar versión (ej: V2_5, v1.0, etc.)
    version_match = re.search(r'[Vv](\d+[._]\d+)', name_without_ext)
    if version_match:
        metadata['version'] = version_match.group(1).replace('_', '.')
    
    return metadata

# Archivos de prueba
test_filenames = [
    "072152-101_CC103_Hardware_en.pdf",
    "AX5000_SystemManual_V2_5.pdf",
    "Lexium32M_UserGuide072022.pdf"
]

print("Prueba de extracción de metadatos de nombres de archivos PDF")
print("=" * 60)

for filename in test_filenames:
    print(f"\nArchivo: {filename}")
    metadata = extract_metadata_from_filename(filename)
    print(f"  Fabricante: {metadata.get('manufacturer', 'No detectado')}")
    print(f"  Modelo: {metadata.get('model', 'No detectado')}")
    print(f"  Versión: {metadata.get('version', 'No detectado')}")

print("\n" + "=" * 60)
print("Prueba completada")