import predict

bind = '0.0.0.0:10080'
workers = 2
timeout = 120

# Reload when code changes detected, 
# helpful during development
reload = True

# Print the error log to stderr,
# helpful during debugging
errorlog = '-'

# Print the access log to stdout,
# helpful during debugging
accesslog = '-'

def on_starting(server):
    predict.load_model()