from .base import router as base_router
from .mnist import router as mnist_router

BASE_ROUTERS = [base_router]
PREDICT_ROUTERS = [mnist_router]
