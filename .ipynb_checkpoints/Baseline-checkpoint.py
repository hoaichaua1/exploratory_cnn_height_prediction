import numpy as np
import matplotlib.pyplot as plt
import glob
import fire
import imageio
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def baseline():

    x = []
    y = []
    images = glob.glob('../synthetic/output/depth*90.png')
    for e, i in enumerate(images):
        txtfl = i.replace('depth', 'scene').replace('.png', '.txt')
        with open(txtfl, "rt") as f:
            label = f.read()
        label = float(label.split(' ')[-1])
        img = imageio.v2.imread(i)[:, :, 0]
        img_min = np.min(img)
        thresh = img_min + 15

        num_pix = np.sum(img < thresh)
        print (num_pix, label)
        x.append(num_pix)
        y.append(label)


    x = np.array(x)
    y = np.array(y)
    lreg = LinearRegression()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    lreg.fit(x_train.reshape(-1, 1), y_train)
    y_pred = lreg.predict(x_test.reshape(-1, 1))


    plt.xlabel("actual child height in cm")
    plt.ylabel("predicted child height in cm")
    plt.title("Baseline model: predicted child height vs actual child height in cm")
    plt.scatter(y_test, y_pred)
    plt.grid(True)
    plt.savefig("baseline_performance.png", dpi=200)

    # let's compute the metrics that we need
    mae = np.sum(np.abs(y_test - y_pred)) / len(y_test)
    pc = np.sum( np.abs(y_test - y_pred) < 2) / len(y_test)

    perc = np.abs(100 - 100 * y_pred / y_test)
    mpe = np.mean(perc)
    print (mpe)

    print (mae, pc, mpe)

if __name__ == '__main__':
    fire.Fire(baseline)