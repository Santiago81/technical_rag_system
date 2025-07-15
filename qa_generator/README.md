# QA Dataset Generator

Sistema de generación automática de datasets de preguntas y respuestas (Q&A) a partir de documentación técnica procesada.

## Descripción

Este módulo genera pares de pregunta-respuesta de alta calidad a partir de chunks de texto almacenados en la base de datos SQLite del sistema RAG. Utiliza modelos de OpenAI para crear preguntas de diferentes tipos y niveles de dificultad, con validación de calidad y filtrado de relevancia.

## Dataset Final Generado

- **Archivo**: `qa_dataset/comprehensive_qa_dataset_final.jsonl`
- **Total QA pairs**: 21,778
- **Tamaño**: 31.2 MB
- **Chunks procesados**: 4,242 de 4,558 (93%)
- **Modelos usados**: gpt-4, gpt-3.5-turbo, gpt-4o-mini
- **Tiempo de procesamiento**: ~48 horas (distribuido en múltiples sesiones)

## Características

- **Múltiples tipos de preguntas**: Factual, Síntesis, Causal, Aplicación, Análisis
- **Niveles de dificultad**: Básico, Intermedio, Avanzado
- **Validación de calidad**: Filtrado automático de preguntas irrelevantes
- **Rate limiting mejorado**: Control adaptativo con backoff exponencial
- **Procesamiento por lotes**: Manejo eficiente de grandes volúmenes
- **Caché de progreso**: Reanudación automática en caso de interrupción
- **Análisis agregado**: Generación de preguntas multi-chunk cada 5 chunks
- **Monitoreo de memoria**: Seguimiento del uso de recursos con psutil

## Instalación

```bash
# Activar entorno virtual
source venv_rag_clean/bin/activate

# Instalar dependencias
pip install langchain-openai langchain-community python-dotenv
```

## Configuración

### Variables de entorno
```bash
# Crear archivo .env
OPENAI_API_KEY=tu_api_key_aqui
```

### Base de datos
El sistema utiliza la base de datos SQLite ubicada en:
```
/Users/santiagojorda/Downloads/clode_technical_rag_system/data/sqlite/manuals.db
```

## Uso

### Procesamiento completo (recomendado)
```bash
# Procesar todos los chunks disponibles
python process_all_chunks_v4.py \
    --model gpt-3.5-turbo \
    --batch-size 3 \
    --rps 2 \
    --output-file comprehensive_qa_dataset_v5.jsonl
```

### Opciones de línea de comandos

- `--model`: Modelo de OpenAI a usar (default: gpt-3.5-turbo)
  - `gpt-3.5-turbo`: Más rápido y económico, límites altos (recomendado)
  - `gpt-4o-mini`: Mejor calidad, límite de 10k requests/día
  - `gpt-4`: Máxima calidad, más costoso, límites bajos
  - `o4-mini`: Nueva versión, requiere temperature=1

- `--batch-size`: Número de chunks por lote (default: 3)
- `--rps`: Límite de solicitudes por segundo (default: 2)
- `--output-file`: Archivo de salida JSONL (default: comprehensive_qa_dataset_v5.jsonl)
- `--save-interval`: Guardar progreso cada N ejemplos (default: 50)
- `--aggregate-interval`: Generar preguntas multi-chunk cada N chunks (default: 5)

### Procesamiento específico de manuales con problemas
```bash
# Analizar y corregir manual específico (ej: CC103)
python fix_cc103_manual.py --analyze --merge --clean
```

## Estructura del output

### Formato JSONL
Cada línea del archivo contiene un objeto JSON con:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Eres un asistente experto en technical documentation..."
    },
    {
      "role": "user",
      "content": "¿Cuál es la función principal del módulo CC10?"
    },
    {
      "role": "assistant",
      "content": "El módulo CC10 es un sistema de control..."
    }
  ],
  "metadata": {
    "source_chunk_ids": [123],
    "source_pdfs": ["manual_cc10.pdf"],
    "page_numbers": [45],
    "difficulty": "intermediate",
    "reasoning_type": "factual",
    "quality_score": 0.85
  }
}
```

## Tipos de preguntas generadas

### 1. Factual (Basic/Intermediate)
- Definiciones y conceptos
- Datos específicos
- Características técnicas

### 2. Synthesis (Intermediate)
- Relaciones entre conceptos
- Integración de información de múltiples fuentes
- Comparaciones

### 3. Causal (Advanced)
- Relaciones causa-efecto
- Análisis de consecuencias
- Razonamiento sobre procesos

### 4. Application (Intermediate)
- Casos de uso prácticos
- Resolución de problemas
- Implementación de conceptos

### 5. Analysis (Advanced)
- Descomposición de sistemas complejos
- Evaluación de componentes
- Análisis crítico

## Monitoreo del progreso

### Ver logs en tiempo real
```bash
tail -f process_chunks_v5_gpt35.log
```

### Verificar progreso específico
```bash
tail -20 process_chunks_v5_gpt35.log | grep -E "Progress:|Examples generated:|Rate limit"
```

### Estadísticas del proceso
```bash
# Ver chunks procesados
grep "Progress:" process_chunks_v5_gpt35.log | tail -1

# Ver ejemplos generados
wc -l qa_dataset/comprehensive_qa_dataset_v5.jsonl

# Ver archivo de progreso
cat processed_chunks_v5.json | jq length
```

### Archivo de progreso
- `processed_chunks_v5.json`: Lista de IDs de chunks ya procesados
- Permite reanudar automáticamente desde el último chunk procesado

## Solución de problemas

### Error 429 (Rate Limit)
- Reducir `--rps` a 2 o menos
- Reducir `--batch-size` a 3 o menos
- Cambiar a otro modelo (cada modelo tiene límites independientes)
- Esperar al reset diario de límites
- El script maneja automáticamente backoff exponencial

### Preguntas irrelevantes
- El sistema filtra automáticamente preguntas genéricas
- Para manuales problemáticos, usar `fix_cc103_manual.py`

### Chunks muy pequeños
- El sistema omite automáticamente chunks < 200 caracteres
- Se pueden fusionar chunks pequeños con el script de corrección

## Archivos importantes

- `qa_generator.py`: Clase principal de generación
- `process_all_chunks_v4.py`: Script de procesamiento masivo con rate limiting mejorado
- `chunk_manager.py`: Gestión de chunks desde la base de datos
- `prompt_templates.py`: Plantillas para diferentes tipos de preguntas
- `quality_evaluator.py`: Evaluación de calidad de Q&A
- `fix_cc103_manual.py`: Corrección de manuales problemáticos
- `processed_chunks_v5.json`: Archivo de progreso para reanudación
- `qa_dataset/comprehensive_qa_dataset_final.jsonl`: Dataset final fusionado

## Rendimiento real observado

- **Velocidad**: ~20-30 ejemplos/minuto con rate limiting conservador
- **Tiempo total**: ~48 horas distribuidas (debido a límites de API)
- **Tasa de procesamiento**: 93% (4,242 de 4,558 chunks)
- **Chunks omitidos**: 316 (muy cortos, <200 caracteres)
- **Output final**: 21,778 pares Q&A de alta calidad
- **Distribución**: ~90% single-chunk, ~10% multi-chunk

## Mejores prácticas

1. **Comenzar con lotes pequeños** (batch-size=3) para verificar calidad
2. **Monitorear los primeros resultados** antes de procesamiento masivo
3. **Usar gpt-3.5-turbo** para balance costo/calidad y límites altos
4. **Guardar logs** para debugging y monitoreo de rate limits
5. **Revisar muestras del output** periódicamente
6. **Usar diferentes modelos** cuando se alcanzan límites diarios
7. **Mantener backups** del archivo de progreso (processed_chunks_v5.json)

## Integración con el sistema RAG

Los datasets generados se pueden usar para:
- Fine-tuning de modelos específicos del dominio
- Evaluación automática del sistema RAG
- Generación de documentación interactiva
- Creación de chatbots especializados

## Licencia

Este módulo es parte del sistema Technical Documentation RAG.