import h5py
path = "dataset/animals/features.hdf5"
db = h5py.File(path)

# when extract_features is completed,
# we get ['features', 'label_names', 'labels']
list(db.keys())

# we get (3000, 25088)
print(db["features"].shape)

# we get (3000,)
print(db["label_names"][0])
