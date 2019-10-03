import redis
import numpy as np
import struct
from PIL import Image
import io

class Mat_Redis_Utils():
    def __init__(self, host='127.0.0.1', port=6379, db=0):
        self.handle = redis.Redis(host, port, db)
        self.dtype_table = [
            np.int8,np.int16,np.int32,np.int64,
            np.uint8,np.uint16,np.uint32,np.uint64,
            np.float16,np.float32,np.float64
        ]
    
    def mat_to_bytes(self, arr):
        dtype_id = self.dtype_table.index(arr.dtype)
        header = struct.pack('>'+'I' * (2+arr.ndim), dtype_id, arr.ndim, *arr.shape)
        data = header + arr.tobytes()
        return data

    def bytes_to_mat(self, data):
        dtype_id, ndim = struct.unpack('>II',data[:8])
        dtype = self.dtype_table[dtype_id]
        shape = struct.unpack('>'+'I'*ndim, data[8:4*(2+ndim)])
        arr = np.frombuffer(data[4*(2+ndim):], dtype=dtype, offset=0)
        arr = arr.reshape((shape))
        return arr
        
    def set(self, key, arr):
        return self.handle.set(key, self.mat_to_bytes(arr))

    def get(self, key, dtype = np.float32):
        data = self.handle.get(key)
        if data is None:
            raise ValueError('%s not exist in Redis'%(key))
        return self.bytes_to_mat(data)

    def set_PIL(self, key, fn):
        return self.handle.set(key, open(fn, "rb").read())

    def get_PIL(self, key):
        data = self.handle.get(key)
        if data is None:
            raise ValueError('%s not exist in Redis'%(key))
        return Image.open(io.BytesIO(data))
    
    def exists(self, key):
        return bool(self.handle.execute_command('EXISTS ' + key))
    
    def ls_keys(self):
        return self.handle.execute_command('KEYS *')
    
    def flush_all(self):
        print('Del all keys in Redis')
        return self.handle.execute_command('flushall')