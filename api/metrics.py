from prometheus_client import Counter, Histogram, Gauge

# Inference counters
OCR_REQUESTS = Counter('ocr_requests_total', 'Total OCR requests processed')
OCR_REJECTS = Counter('ocr_rejects_total', 'Total OCR requests rejected to QC')

# Accuracy Tracking (Updated via background evaluation jobs or sampled GT)
DIGIT_ACCURACY = Gauge('ocr_digit_accuracy', 'Latest sampled digit accuracy')
FIELD_ACCURACY = Gauge('ocr_field_accuracy', 'Latest sampled field accuracy')
CER = Gauge('ocr_cer', 'Character Error Rate')

# Latency
INFERENCE_LATENCY = Histogram('ocr_inference_latency_seconds', 'Latency of /infer endpoint')

def record_inference_metrics(latency_sec: float, rejected: bool):
    OCR_REQUESTS.inc()
    if rejected:
        OCR_REJECTS.inc()
    INFERENCE_LATENCY.observe(latency_sec)
