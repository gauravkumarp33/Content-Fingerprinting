from app.services.vector_store import add_embedding, search
import numpy as np

vec = np.random.rand(512).astype('float32')
add_embedding("1", vec)

results = search(vec, 5)
print(results)