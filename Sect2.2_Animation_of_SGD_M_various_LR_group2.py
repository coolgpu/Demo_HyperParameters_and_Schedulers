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
             'WbStr', 'Wb', 'uvWaStr', 'unitvecWa', 'uvWbStr', 'unitvecWb', 'stepsizeStr', 'eff_stepsize',
             'dWaStr','dWa','dWbStr','dWb','VdWaStr','VdWa','VdWbStr','VdWb','SdWaStr','SdWa','SdWbStr','SdWb']
    df = pd.read_csv(filename, names=names, skiprows=3, skipfooter=2, engine='python', sep='[,=]')
    WaTraces = df['Wa'].values
    WbTraces = df['Wb'].values
    LossTraces = df['loss'].values
    EpochTraces = df['Epoch'].values
    minibatchTraces = df['minibatch'].values
    updateTraces = df['update'].values
    stepsizeTrances = df['eff_stepsize'].values
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
        lr = line2.split()[-1]
        # line3 = f.readline()
        # _, wa, wb = line3.split()

    return WaTraces, WbTraces, LossTraces, EpochTraces, minibatchTraces, updateTraces, lr, method, \
    stepsizeTrances, unitvecWaTraces, unitvecWbTraces, dWaTraces, dWbTraces, VdWaTraces, VdWbTraces, SdWaTraces, SdWbTraces


def main():
    device = torch.device('cpu')
    torch.manual_seed(9999)
    a = 1.261845
    b = 1.234378
    c = math.sqrt(a * a - b * b)
    nsamples = 512
    batch_size = 512

    # load previously generated results
    WaTraces_SGD_MOMENTUM1, WbTraces_SGD_MOMENTUM1, LossTraces_SGD_MOMENTUM1, EpochTraces1, minibatchTraces1, updateTraces1, lr_SGD_MOMENTUM1, method_SGD_MOMENTUM1, \
        _, _, _, dWa_SGD_MOMENTUM1, dWb_SGD_MOMENTUM1, _, _, _, _ = \
        read_optimizer_results(r'results/Sect2.2_SGD_Momentum_lr0.012_Epoch400_results.log')
    WaTraces_SGD_MOMENTUM2, WbTraces_SGD_MOMENTUM2, LossTraces_SGD_MOMENTUM2, EpochTraces2, minibatchTraces2, updateTraces2,  lr_SGD_MOMENTUM2, method_SGD_MOMENTUM2, \
        _, _, _, dWa_SGD_MOMENTUM2, dWb_SGD_MOMENTUM2, _, _, _, _ = \
        read_optimizer_results(r'results/Sect2.2_SGD_Momentum_lr0.012_Epoch100_results.log')
    WaTraces_SGD_MOMENTUM3, WbTraces_SGD_MOMENTUM3, LossTraces_SGD_MOMENTUM3, EpochTraces3, minibatchTraces3, updateTraces3,  lr_SGD_MOMENTUM3, method_SGD_MOMENTUM3, \
        _, _, _, dWa_SGD_MOMENTUM3, dWb_SGD_MOMENTUM3, _, _, _, _ = \
        read_optimizer_results(r'results/Sect2.2_SGD_Momentum_lr0.00015_Epoch400_results.log')
    WaTraces_SGD_MOMENTUM4, WbTraces_SGD_MOMENTUM4, LossTraces_SGD_MOMENTUM4, EpochTraces4, minibatchTraces4, updateTraces4,  lr_SGD_MOMENTUM4, method_SGD_MOMENTUM4, \
        _, _, _, dWa_SGD_MOMENTUM4, dWb_SGD_MOMENTUM4, _, _, _, _ = \
        read_optimizer_results(r'results/Sect2.2_SGD_Momentum_lr0.00015_Epoch100_results.log')

    nframes = max([WaTraces_SGD_MOMENTUM1.size, WaTraces_SGD_MOMENTUM2.size, WaTraces_SGD_MOMENTUM3.size, WaTraces_SGD_MOMENTUM4.size])
    epoch1, epoch2, epoch3, epoch4 = EpochTraces1[-1]+1, EpochTraces2[-1]+1, EpochTraces3[-1]+1, EpochTraces4[-1] + 1
    update1, update2, update3, update4 = updateTraces1[-1], updateTraces2[-1], updateTraces3[-1], updateTraces4[-1]

    xy_dataset = EllipseDataset(nsamples, a, b, noise_scale=0.1)
    xy_dataloader = DataLoader(xy_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    Wa0 = WaTraces_SGD_MOMENTUM1[0]
    Wb0 = WbTraces_SGD_MOMENTUM1[0]

    # nWaGrids, nWbGrids = 200, 200
    # WaGrid = np.linspace(0, 2.0, nWaGrids)
    # WbGrid = np.linspace(0, 2.0, nWbGrids)
    nWaGrids, nWbGrids = 250, 250
    # WaGrid = np.linspace(0, 2.5, nWaGrids)
    # WbGrid = np.linspace(0, 2.5, nWbGrids)
    WaGrid = np.linspace(0, 4.0, nWaGrids)
    WbGrid = np.linspace(-0.5, 3.5, nWbGrids)

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
    ax.set(xlim=(0, 4.0), ylim=(-0.5, 3.5))
    ax.set_title('SGD_w_Momentum Training Progress w/ More Epochs', fontsize=16)
    plt.xlabel("Wa")
    plt.ylabel("Wb")
    ax.set_aspect('equal', adjustable='box')
    ax.plot(a, b, 'yo', ms=4)
    ax.plot(Wa0, Wb0, 'ko', ms=5)
    ax.text(0.08, 1.85, 'Start', color="white", fontsize=14)
    ax.text(0.88, 1.15, 'Target', color="white", fontsize=14)

    walist_SGD_MOMENTUM1 = []
    wblist_SGD_MOMENTUM1 = []
    point_SGD_MOMENTUM1, = ax.plot([], [], 'ro', lw=0.5, markersize=5)
    line_SGD_MOMENTUM1, = ax.plot([], [], '-r', lw=3, label='SGD_Momentum epoch={} '.format(epoch1)+lr_SGD_MOMENTUM1)

    walist_SGD_MOMENTUM2 = []
    wblist_SGD_MOMENTUM2 = []
    point_SGD_MOMENTUM2, = ax.plot([], [], 'yo', lw=0.5, markersize=3)
    line_SGD_MOMENTUM2, = ax.plot([], [], '-y', lw=1, label='SGD_Momentum epoch={} '.format(epoch2)+lr_SGD_MOMENTUM2)

    walist_SGD_MOMENTUM3 = []
    wblist_SGD_MOMENTUM3 = []
    point_SGD_MOMENTUM3, = ax.plot([], [], 'mo', lw=0.5, markersize=5)
    line_SGD_MOMENTUM3, = ax.plot([], [], '-m', lw=3, label='SGD_Momentum epoch={} '.format(epoch3)+lr_SGD_MOMENTUM3)

    walist_SGD_MOMENTUM4 = []
    wblist_SGD_MOMENTUM4 = []
    point_SGD_MOMENTUM4, = ax.plot([], [], 'o', lw=0.5, markersize=3, color='aqua')
    line_SGD_MOMENTUM4, = ax.plot([], [], '-', lw=1, color='aqua', label='SGD_Momentum epoch={} '.format(epoch4)+lr_SGD_MOMENTUM4)

    text_update = ax.text(0.03, 0.03, '', transform=ax.transAxes, color="white", fontsize=14)

    leg = ax.legend()
    fig.tight_layout()
    plt.show(block=False)

    # initialization function: plot the background of each frame
    def init():
        point_SGD_MOMENTUM1.set_data([], [])
        line_SGD_MOMENTUM1.set_data([], [])

        point_SGD_MOMENTUM2.set_data([], [])
        line_SGD_MOMENTUM2.set_data([], [])

        point_SGD_MOMENTUM3.set_data([], [])
        line_SGD_MOMENTUM3.set_data([], [])

        point_SGD_MOMENTUM4.set_data([], [])
        line_SGD_MOMENTUM4.set_data([], [])

        text_update.set_text('')

        return point_SGD_MOMENTUM1, line_SGD_MOMENTUM1, point_SGD_MOMENTUM2, line_SGD_MOMENTUM2, point_SGD_MOMENTUM3, line_SGD_MOMENTUM3, point_SGD_MOMENTUM4, line_SGD_MOMENTUM4, text_update

    # animation function.  This is called sequentially
    def animate(i):
        if i == 0:

            wblist_SGD_MOMENTUM1[:] = []
            walist_SGD_MOMENTUM1[:] = []

            wblist_SGD_MOMENTUM2[:] = []
            walist_SGD_MOMENTUM2[:] = []

            wblist_SGD_MOMENTUM3[:] = []
            walist_SGD_MOMENTUM3[:] = []

            wblist_SGD_MOMENTUM4[:] = []
            walist_SGD_MOMENTUM4[:] = []

        if i < update1:
            wa_SGD_MOMENTUM1, wb_SGD_MOMENTUM1 = WaTraces_SGD_MOMENTUM1[i], WbTraces_SGD_MOMENTUM1[i]
            wblist_SGD_MOMENTUM1.append(wa_SGD_MOMENTUM1)
            walist_SGD_MOMENTUM1.append(wb_SGD_MOMENTUM1)
            point_SGD_MOMENTUM1.set_data(wa_SGD_MOMENTUM1, wb_SGD_MOMENTUM1)
            line_SGD_MOMENTUM1.set_data(wblist_SGD_MOMENTUM1, walist_SGD_MOMENTUM1)

        if i < update2:
            wa_SGD_MOMENTUM2, wb_SGD_MOMENTUM2  = WaTraces_SGD_MOMENTUM2[i], WbTraces_SGD_MOMENTUM2[i]
            wblist_SGD_MOMENTUM2.append(wa_SGD_MOMENTUM2)
            walist_SGD_MOMENTUM2.append(wb_SGD_MOMENTUM2)
            point_SGD_MOMENTUM2.set_data(wa_SGD_MOMENTUM2, wb_SGD_MOMENTUM2)
            line_SGD_MOMENTUM2.set_data(wblist_SGD_MOMENTUM2, walist_SGD_MOMENTUM2)

        if i < update3:
            wa_SGD_MOMENTUM3, wb_SGD_MOMENTUM3 = WaTraces_SGD_MOMENTUM3[i], WbTraces_SGD_MOMENTUM3[i]
            wblist_SGD_MOMENTUM3.append(wa_SGD_MOMENTUM3)
            walist_SGD_MOMENTUM3.append(wb_SGD_MOMENTUM3)
            point_SGD_MOMENTUM3.set_data(wa_SGD_MOMENTUM3, wb_SGD_MOMENTUM3)
            line_SGD_MOMENTUM3.set_data(wblist_SGD_MOMENTUM3, walist_SGD_MOMENTUM3)

        if i < update4:
            wa_SGD_MOMENTUM4, wb_SGD_MOMENTUM4 = WaTraces_SGD_MOMENTUM4[i], WbTraces_SGD_MOMENTUM4[i]
            wblist_SGD_MOMENTUM4.append(wa_SGD_MOMENTUM4)
            walist_SGD_MOMENTUM4.append(wb_SGD_MOMENTUM4)
            point_SGD_MOMENTUM4.set_data(wa_SGD_MOMENTUM4, wb_SGD_MOMENTUM4)
            line_SGD_MOMENTUM4.set_data(wblist_SGD_MOMENTUM4, walist_SGD_MOMENTUM4)
        
        update, epoch, minibatch = updateTraces1[i], EpochTraces1[i]+1, minibatchTraces1[i]+1
        text_update.set_text('Epoch={:d}, minibatch={:d}, Updates={:d}'.format(epoch, minibatch, update))

        return point_SGD_MOMENTUM1, line_SGD_MOMENTUM1, point_SGD_MOMENTUM2, line_SGD_MOMENTUM2, point_SGD_MOMENTUM3, line_SGD_MOMENTUM3, point_SGD_MOMENTUM4, line_SGD_MOMENTUM4, text_update

    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    intervalms = 10  # this means 10 ms per frame
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nframes, interval=intervalms, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be installed.
    # anim.save('Results/Part5_Fig12_Animation_SGD_Momentum_increasedEpoch.mp4', fps=30, bitrate=1800)
    plt.show()

    # extra: plot the gradients of Wa and Wb for the case of LR=0.012 and epochs=400
    flag_generate_Fig13 = True
    if flag_generate_Fig13:
        fig = plt.figure(figsize=(12, 6))
        r = range(35, 1635)
        plt.plot(r, dWb_SGD_MOMENTUM1[r], '-', label='Gradient of Wb', lw=1)
        plt.plot(r, dWa_SGD_MOMENTUM1[r], '-r', label='Gradient of Wa', lw=1)
        plt.xlim(35, 1635)
        plt.xlabel("Number of Updates during Training")
        plt.ylabel("Gradient of Loss w.r.t. Wa or Wb")
        plt.savefig(r'Results/Part5_Fig13_Demonstrate_effect_of_momentum.png')
        plt.show()
    # -------------------------------------------

    print('Done!')

if __name__ == '__main__':
    main()

