import numpy as np
from PIL import Image
import glob

for path in glob.glob("results/*npy"):

    npFile = np.load(path)

    if len(npFile.shape) == 4:
        npFile = npFile[0].transpose(1,2,0)
    else:
        if npFile.shape[0] == 1:
            npFile = npFile[0]

    if npFile.dtype != "uint8":
        npFile = npFile.astype("uint8")

    if npFile.shape[0] == 4:
        for i in range(npFile.shape[0]):
            print(npFile[i].shape,npFile[i].dtype)
            im = Image.fromarray(npFile[i])
            im.save(path.replace(".npy","_{}.png".format(i)))

    else:
        print(npFile.shape,npFile.dtype)
        im = Image.fromarray(npFile)
        im.save(path.replace("npy","png"))
