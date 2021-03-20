import torch
from torch.utils.data import DataLoader
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from Ellipse_Dataset import EllipseDataset
from torch import optim
import pandas as pd

def read_optimizer_results(filename):
    names = ['updateStr', 'update', 'EpochStr', 'Epoch', 'minibatchStr', 'minibatch', 'lossStr', 'loss', 'WaStr', 'Wa',
             'WbStr', 'Wb', 'uvWaStr', 'unitvecWa', 'uvWbStr', 'unitvecWb', 'lrStr', 'learningrate',
             'dWaStr','dWa','dWbStr','dWb','VdWaStr','VdWa','VdWbStr','VdWb','SdWaStr','SdWa','SdWbStr','SdWb']
    df = pd.read_csv(filename, names=names, skiprows=3, skipfooter=2, engine='python', sep='[,=]')
    WaTraces = df['Wa'].values
    WbTraces = df['Wb'].values
    LossTraces = df['loss'].values
    EpochTraces = df['Epoch'].values
    minibatchTraces = df['minibatch'].values
    updateTraces = df['update'].values
    lrTrances = df['learningrate'].values
    unitvecWaTraces = df['unitvecWa'].values
    unitvecWbTraces = df['unitvecWb'].values
    dWaTraces = df['dWa'].values
    dWbTraces = df['dWb'].values
    VdWaTraces = df['VdWa'].values
    VdWbTraces = df['VdWb'].values
    SdWaTraces = df['SdWa'].values
    SdWbTraces = df['SdWb'].values

    # get the meta data in the first 3 lines
    with open(filename, "r") as f:
        line1 = f.readline()
        method = line1.split()[0]
        line2 = f.readline()
        initLR, lr_gamma = line2.split(', ')[-2], line2.split(', ')[-1]
        # line3 = f.readline()
        # _, wa, wb = line3.split()

    return WaTraces, WbTraces, LossTraces, EpochTraces, minibatchTraces, updateTraces, initLR, lr_gamma, method, \
    lrTrances, unitvecWaTraces, unitvecWbTraces, dWaTraces, dWbTraces, VdWaTraces, VdWbTraces, SdWaTraces, SdWbTraces


def main():
    device = torch.device('cpu')
    torch.manual_seed(9999)
    a = 1.261845
    b = 1.234378
    c = math.sqrt(a * a - b * b)
    nsamples = 512
    batch_size = 512

    # load previously generated results
    WaTraces_Adam1, WbTraces_Adam1, LossTraces_Adam1, EpochTraces1, minibatchTraces1, updateTraces1, lr_Adam1, gamma_Adam1, method_Adam1, \
        lrTraces_Adam1, _, _, dWa_Adam1, dWb_Adam1, _, _, _, _ = \
        read_optimizer_results(r'results/Adam_lr0.01_Epoch100_Schl1.00_results.log')
    WaTraces_Adam2, WbTraces_Adam2, LossTraces_Adam2, EpochTraces2, minibatchTraces2, updateTraces2,  lr_Adam2, gamma_Adam2, method_Adam2, \
        lrTraces_Adam2, _, _, dWa_Adam2, dWb_Adam2, _, _, _, _ = \
        read_optimizer_results(r'results/Adam_lr0.12_Epoch100_Schl1.00_results.log')
    WaTraces_Adam3, WbTraces_Adam3, LossTraces_Adam3, EpochTraces3, minibatchTraces3, updateTraces3,  lr_Adam3, gamma_Adam3, method_Adam3, \
        lrTraces_Adam3, _, _, dWa_Adam3, dWb_Adam3, _, _, _, _ = \
        read_optimizer_results(r'results/Adam_lr0.12_Epoch100_Schl_lambda_fn_0_8_results.log')
    WaTraces_Adam4, WbTraces_Adam4, LossTraces_Adam4, EpochTraces4, minibatchTraces4, updateTraces4,  lr_Adam4, gamma_Adam4, method_Adam4, \
        lrTraces_Adam4, _, _, dWa_Adam4, dWb_Adam4, _, _, _, _ = \
        read_optimizer_results(r'results/Adam_lr0.12_Epoch100_Schl_lambda_fn_0_5_results.log')

    nframes = max([WaTraces_Adam1.size, WaTraces_Adam2.size, WaTraces_Adam3.size, WaTraces_Adam4.size])
    epoch1, epoch2, epoch3, epoch4 = EpochTraces1[-1]+1, EpochTraces2[-1]+1, EpochTraces3[-1]+1, EpochTraces4[-1] + 1
    update1, update2, update3, update4 = updateTraces1[-1], updateTraces2[-1], updateTraces3[-1], updateTraces4[-1]

    xy_dataset = EllipseDataset(nsamples, a, b, noise_scale=0.1)
    xy_dataloader = DataLoader(xy_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    Wa0 = WaTraces_Adam1[0]
    Wb0 = WbTraces_Adam1[0]

    # nWaGrids, nWbGrids = 200, 200
    # WaGrid = np.linspace(0, 2.0, nWaGrids)
    # WbGrid = np.linspace(0, 2.0, nWbGrids)
    nWaGrids, nWbGrids = 200, 200
    # WaGrid = np.linspace(0, 2.5, nWaGrids)
    # WbGrid = np.linspace(0, 2.5, nWbGrids)
    WaGrid = np.linspace(0, 2.0, nWaGrids)
    WbGrid = np.linspace(0.25, 2.25, nWbGrids)

    Wa2d, Wb2d = np.meshgrid(WaGrid, WaGrid)
    loss = np.zeros(Wa2d.shape)

    for i_batch, sample_batched in enumerate(xy_dataloader):
        x, y = sample_batched['x'], sample_batched['y']

    for indexb, Wb in enumerate(WbGrid):
        for indexa, Wa in enumerate(WaGrid):
            y_pred_sqr = Wb ** 2 * (1.0 - (x + c) ** 2 / Wa ** 2)
            y_pred_sqr[y_pred_sqr < 0.00000001] = 0.00000001  # handle negative values caused by noise
            y_pred = torch.sqrt(y_pred_sqr)

            loss[indexb, indexa] = (y_pred - y).pow(2).sum()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.contour(WbGrid, WaGrid, loss, levels=15, linewidths=0.5, colors='gray')
    cntr1 = ax.contourf(WaGrid, WbGrid, loss, levels=100, cmap="RdBu_r")
    fig.colorbar(cntr1, ax=ax, shrink=0.75)
    # ax.set(xlim=(0, 2.0), ylim=(0, 2.0))
    ax.set(xlim=(0, 2), ylim=(0.25, 2.25))
    ax.set_title('Adam Optimization with MultiplicativeLR Schedulers & Lambda', fontsize=16)
    plt.xlabel("Wa")
    plt.ylabel("Wb")
    ax.set_aspect('equal', adjustable='box')
    ax.plot(a, b, 'yo', ms=5)
    ax.plot(Wa0, Wb0, 'ko', ms=5)
    ax.text(0.08, 1.85, 'Start', color="white", fontsize=14)
    ax.text(0.88, 1.15, 'Target', color="white", fontsize=14)

    walist_Adam1 = []
    wblist_Adam1 = []
    point_Adam1, = ax.plot([], [], 'ro', lw=0.5, markersize=4)
    line_Adam1, = ax.plot([], [], '-r', lw=1, label='Adam epoch={}, '.format(epoch1)+lr_Adam1+' fixed')

    walist_Adam2 = []
    wblist_Adam2 = []
    point_Adam2, = ax.plot([], [], 'yo', lw=0.5, markersize=4)
    line_Adam2, = ax.plot([], [], '-y', lw=1, label='Adam epoch={}, '.format(epoch2)+lr_Adam2+' fixed')

    walist_Adam3 = []
    wblist_Adam3 = []
    point_Adam3, = ax.plot([], [], 'mo', lw=0.5, markersize=4)
    line_Adam3, = ax.plot([], [], '-m', lw=1, label='Adam epoch={}, '.format(epoch3)+lr_Adam3+' MultiplicativeLR w/ Lambda fn 0_8')

    walist_Adam4 = []
    wblist_Adam4 = []
    point_Adam4, = ax.plot([], [], 'o', lw=0.5, markersize=4, color='aqua')
    line_Adam4, = ax.plot([], [], '-', lw=1, color='aqua', label='Adam epoch={}, '.format(epoch4)+lr_Adam4+' MultiplicativeLR w/ Lambda fn 0_5')

    text_update = ax.text(0.03, 0.03, '', transform=ax.transAxes, color="blue", fontsize=14)

    leg = ax.legend()
    fig.tight_layout()
    plt.show(block=False)

    # initialization function: plot the background of each frame
    def init():
        point_Adam1.set_data([], [])
        line_Adam1.set_data([], [])

        point_Adam2.set_data([], [])
        line_Adam2.set_data([], [])

        point_Adam3.set_data([], [])
        line_Adam3.set_data([], [])

        point_Adam4.set_data([], [])
        line_Adam4.set_data([], [])

        text_update.set_text('')

        return point_Adam1, line_Adam1, point_Adam2, line_Adam2, point_Adam3, line_Adam3, point_Adam4, line_Adam4, text_update

    # animation function.  This is called sequentially
    def animate(i):
        if i == 0:

            wblist_Adam1[:] = []
            walist_Adam1[:] = []

            wblist_Adam2[:] = []
            walist_Adam2[:] = []

            wblist_Adam3[:] = []
            walist_Adam3[:] = []

            wblist_Adam4[:] = []
            walist_Adam4[:] = []

        if i < update1:
            wa_Adam1, wb_Adam1 = WaTraces_Adam1[i], WbTraces_Adam1[i]
            wblist_Adam1.append(wa_Adam1)
            walist_Adam1.append(wb_Adam1)
            point_Adam1.set_data(wa_Adam1, wb_Adam1)
            line_Adam1.set_data(wblist_Adam1, walist_Adam1)

        if i < update2:
            wa_Adam2, wb_Adam2  = WaTraces_Adam2[i], WbTraces_Adam2[i]
            wblist_Adam2.append(wa_Adam2)
            walist_Adam2.append(wb_Adam2)
            point_Adam2.set_data(wa_Adam2, wb_Adam2)
            line_Adam2.set_data(wblist_Adam2, walist_Adam2)

        if i < update3:
            wa_Adam3, wb_Adam3 = WaTraces_Adam3[i], WbTraces_Adam3[i]
            wblist_Adam3.append(wa_Adam3)
            walist_Adam3.append(wb_Adam3)
            point_Adam3.set_data(wa_Adam3, wb_Adam3)
            line_Adam3.set_data(wblist_Adam3, walist_Adam3)

        if i < update4:
            wa_Adam4, wb_Adam4 = WaTraces_Adam4[i], WbTraces_Adam4[i]
            wblist_Adam4.append(wa_Adam4)
            walist_Adam4.append(wb_Adam4)
            point_Adam4.set_data(wa_Adam4, wb_Adam4)
            line_Adam4.set_data(wblist_Adam4, walist_Adam4)
        
        update, epoch, minibatch = updateTraces1[i], EpochTraces1[i]+1, minibatchTraces1[i]+1
        text_update.set_text('Epoch={:d}, minibatch={:d}, Updates={:d}'.format(epoch, minibatch, update))

        return point_Adam1, line_Adam1, point_Adam2, line_Adam2, point_Adam3, line_Adam3, point_Adam4, line_Adam4, text_update

    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    intervalms = 10  # this means 10 ms per frame
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nframes, interval=intervalms, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be installed.
    anim.save(r'results/Animation_Adam_Scheduler3.3.mp4', fps=30, bitrate=1800)

    plt.show()

    print('Done!')

if __name__ == '__main__':
    main()

