import numpy as np
from annoy import AnnoyIndex

t = AnnoyIndex(250, "dot")

for i in range(1000):
    t.add_item(i, np.absolute(np.random.rand(250)))

# segfault
t.build(16)
t.build(16)
t.build(16)
t.save("test.ann")
print(t.get_nns_by_item(0, 10))

