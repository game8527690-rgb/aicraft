import multiprocessing

# Workers
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 180  # 3 min - needed for image generation

# Server
bind = "0.0.0.0:5000"
forwarded_allow_ips = "*"
proxy_protocol = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Security
limit_request_line = 4096
limit_request_fields = 100
