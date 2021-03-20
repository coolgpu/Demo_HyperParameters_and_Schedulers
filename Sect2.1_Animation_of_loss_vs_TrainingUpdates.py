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
    WaTraces_Adam, WbTraces_Adam, LossTraces_Adam, EpochTraces, minibatchTraces, updateTraces, lr_Adam, method_Adam, \
        stepsize_Adam, unitvecWa_Adam, unitvecWb_Adam, dWa_Adam, dWb_Adam, VdWa_Adam, VdWb_Adam, SdWa_Adam, SdWb_Adam = \
        read_optimizer_results(r'results/Adam_custom_implement_LR0.01_results.log')
    WaTraces_SGD, WbTraces_SGD, LossTraces_SGD, _, _, _, lr_SGD, method_SGD,\
        stepsize_SGD, unitvecWa_SGD, unitvecWb_SGD, dWa_SGD, dWb_SGD, _, _, _, _ = \
        read_optimizer_results(r'results/SGD_custom_implement_LR0.01_results.log')
    WaTraces_SGD_MOMENTUM, WbTraces_SGD_MOMENTUM, LossTraces_SGD_MOMENTUM, _, _, _, lr_SGD_MOMENTUM, method_SGD_MOMENTUM, \
        stepsize_SGD_MOMENTUM, unitvecWa_SGD_MOMENTUM, unitvecWb_SGD_MOMENTUM, dWa_SGD_MOMENTUM, dWb_SGD_MOMENTUM, VdWa_SGD_MOMENTUM, VdWa_SGD_MOMENTUM, _, _ = \
        read_optimizer_results(r'results/SGD_w_Momentum_custom_implement_LR0.01_results.log')
    WaTraces_RMSprop, WbTraces_RMSprop, LossTraces_RMSprop, _, _, _, lr_RMSprop, method_RMSprop, \
        stepsize_RMSprop, unitvecWa_RMSprop, unitvecWb_RMSprop, dWa_RMSprop, dWb_RMSprop, VdWa_RMSprop, VdWb_RMSprop, SdWa_RMSprop, SdWb_RMSprop = \
        read_optimizer_results(r'results/RMSprop_custom_implement_LR0.01_results.log')


    # First plot: stepsize vs update ------------
    if True:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        line_SGD, = ax.plot(updateTraces, stepsize_SGD, '-y', lw=1, label=method_SGD+' '+lr_SGD)
        line_SGD_MOMENTUM, = ax.plot(updateTraces, stepsize_SGD_MOMENTUM, '-r', lw=1, label=method_SGD_MOMENTUM+' '+lr_SGD_MOMENTUM)
        line_RMSprop, = ax.plot(updateTraces, stepsize_RMSprop, '-b', lw=0.5, label=method_RMSprop+' '+lr_RMSprop)
        line_Adam, = ax.plot(updateTraces, stepsize_Adam, '-g', lw=1, label=method_Adam+' '+lr_Adam)
        leg = ax.legend()
        ax.set(xlim=(0, 600), ylim=(0, 0.01+max(stepsize_SGD.max(), stepsize_SGD_MOMENTUM.max(), stepsize_RMSprop.max(), stepsize_Adam.max())))
        ax.set_title('Effective Step size as a function of Updates', fontsize=16)
        plt.xlabel("# of Updates")
        plt.ylabel("Effective Step Size (AU)")
        fig.tight_layout()
        plt.savefig(r'results/Part5_Fig1_EffStepSize_vs_Updates.png')
        plt.show(block=False)
    # -------------------------------------------




    nframes = len(WaTraces_Adam)
    # nframes = 600  # len(WaTraces_Adam)

    xy_dataset = EllipseDataset(nsamples, a, b, noise_scale=0.1)
    xy_dataloader = DataLoader(xy_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    Wa0 = WaTraces_Adam[0]
    Wb0 = WbTraces_Adam[0]

    # nWaGrids, nWbGrids = 200, 200
    # WaGrid = np.linspace(0, 2.0, nWaGrids)
    # WbGrid = np.linspace(0, 2.0, nWbGrids)
    nWaGrids, nWbGrids = 250, 250
    WaGrid = np.linspace(0, 2.5, nWaGrids)
    WbGrid = np.linspace(0, 2.5, nWbGrids)
    # WaGrid = np.linspace(0, 0.5, nWaGrids) 
    # WbGrid = np.linspace(1.5, 2.0, nWbGrids)

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

    # Second plot: plot Steps during the 1st few updates ------------
    if True:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.contour(WbGrid, WaGrid, loss, levels=15, linewidths=0.5, colors='gray')
        cntr1 = ax.contourf(WaGrid, WbGrid, loss, levels=100, cmap="RdBu_r")
        fig.colorbar(cntr1, ax=ax, shrink=0.75)
        # ax.set(xlim=(0, 2.0), ylim=(0, 2.0))
        # ax.set(xlim=(0, 2.5), ylim=(0, 2.5))
        ax.set(xlim=(0, 0.5), ylim=(1.5, 2.0))
        ax.set_title('First 4 updates of Wa and Wb in training', fontsize=16)
        plt.xlabel("Wa")
        plt.ylabel("Wb")
        ax.set_aspect('equal', adjustable='box')
        ax.plot(a, b, 'yo', ms=3)
        ax.plot(Wa0, Wb0, 'ko', ms=3)
        ax.text(0.08, 1.81, 'Start', color="black", fontsize=14)

        for i in range(4):
            dWa_SGD = (WaTraces_SGD[i+1]-WaTraces_SGD[i]) * 0.995
            dWb_SGD = (WbTraces_SGD[i+1]-WbTraces_SGD[i]) * 0.995
            plt.arrow(WaTraces_SGD[i], WbTraces_SGD[i], dWa_SGD, dWb_SGD, color='yellow', linewidth=1, head_width=0.005, length_includes_head=True)

            dWa_SGD_M = (WaTraces_SGD_MOMENTUM[i + 1] - WaTraces_SGD_MOMENTUM[i]) * 0.995
            dWb_SGD_M = (WbTraces_SGD_MOMENTUM[i + 1] - WbTraces_SGD_MOMENTUM[i]) * 0.995
            plt.arrow(WaTraces_SGD_MOMENTUM[i], WbTraces_SGD_MOMENTUM[i], dWa_SGD_M, dWb_SGD_M, color='red', linewidth=1, head_width=0.005, length_includes_head=True)

            dWa_RMSprop = (WaTraces_RMSprop[i + 1] - WaTraces_RMSprop[i]) * 0.995
            dWb_RMSprop = (WbTraces_RMSprop[i + 1] - WbTraces_RMSprop[i]) * 0.995
            plt.arrow(WaTraces_RMSprop[i], WbTraces_RMSprop[i], dWa_RMSprop, dWb_RMSprop, color='blue', linewidth=1, head_width=0.005, length_includes_head=True)

            dWa_Adam = (WaTraces_Adam[i + 1] - WaTraces_Adam[i]) * 0.995
            dWb_Adam = (WbTraces_Adam[i + 1] - WbTraces_Adam[i]) * 0.995
            plt.arrow(WaTraces_Adam[i], WbTraces_Adam[i], dWa_Adam, dWb_Adam, color='green', linewidth=1, head_width=0.005, length_includes_head=True)

        line_SGD, = ax.plot([], [], '-y', lw=1, label=method_SGD + ' ' + lr_SGD)
        line_SGD_MOMENTUM, = ax.plot([], [], '-r', lw=1, label=method_SGD_MOMENTUM + ' ' + lr_SGD_MOMENTUM)
        line_RMSprop, = ax.plot([], [], '-b', lw=0.5, label=method_RMSprop + ' ' + lr_RMSprop)
        line_Adam, = ax.plot([], [], '-g', lw=1, label=method_Adam + ' ' + lr_Adam)

        leg = ax.legend()
        fig.tight_layout()
        plt.savefig('results/Part5_Fig2_FirstFewSteps.png')
        plt.show(block=False)
    # -------------------------------------------

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.contour(WbGrid, WaGrid, loss, levels=15, linewidths=0.5, colors='gray')
    cntr1 = ax.contourf(WaGrid, WbGrid, loss, levels=100, cmap="RdBu_r")
    fig.colorbar(cntr1, ax=ax, shrink=0.75)
    # ax.set(xlim=(0, 2.0), ylim=(0, 2.0))
    ax.set(xlim=(0, 2.5), ylim=(0, 2.5))
    ax.set_title('Different Optimization Paths Using The Same Learning Rate', fontsize=16)
    plt.xlabel("Wa")
    plt.ylabel("Wb")
    ax.set_aspect('equal', adjustable='box')
    ax.plot(a, b, 'yo', ms=3)
    ax.plot(Wa0, Wb0, 'ko', ms=3)

    wblist_SGD = []
    walist_SGD = []
    point_SGD, = ax.plot([], [], 'yo', lw=0.5, markersize=4)
    line_SGD, = ax.plot([], [], '-y', lw=2, label=method_SGD+' '+lr_SGD)

    wblist_SGD_MOMENTUM = []
    walist_SGD_MOMENTUM = []
    point_SGD_MOMENTUM, = ax.plot([], [], 'ro', lw=0.5, markersize=4)
    line_SGD_MOMENTUM, = ax.plot([], [], '-r', lw=2, label=method_SGD_MOMENTUM+' '+lr_SGD_MOMENTUM)

    wblist_RMSprop = []
    walist_RMSprop = []
    point_RMSprop, = ax.plot([], [], 'mo', lw=0.5, markersize=4)
    line_RMSprop, = ax.plot([], [], '-m', lw=2, label=method_RMSprop+' '+lr_RMSprop)

    wblist_Adam = []
    walist_Adam = []
    point_Adam, = ax.plot([], [], 'go', lw=0.5, markersize=4)
    line_Adam, = ax.plot([], [], '-g', lw=2, label=method_Adam+' '+lr_Adam)

    text_update = ax.text(0.03, 0.03, '', transform=ax.transAxes, color="black", fontsize=14)

    leg = ax.legend()
    fig.tight_layout()
    plt.show(block=False)

    # initialization function: plot the background of each frame
    def init():
        point_SGD.set_data([], [])
        line_SGD.set_data([], [])

        point_SGD_MOMENTUM.set_data([], [])
        line_SGD_MOMENTUM.set_data([], [])

        point_RMSprop.set_data([], [])
        line_RMSprop.set_data([], [])

        point_Adam.set_data([], [])
        line_Adam.set_data([], [])

        text_update.set_text('')

        return point_SGD, line_SGD, point_SGD_MOMENTUM, line_SGD_MOMENTUM, point_RMSprop, line_RMSprop, point_Adam, line_Adam, text_update

    # animation function.  This is called sequentially
    def animate(i):
        if i == 0:

            wblist_SGD[:] = []
            walist_SGD[:] = []

            wblist_SGD_MOMENTUM[:] = []
            walist_SGD_MOMENTUM[:] = []

            wblist_RMSprop[:] = []
            walist_RMSprop[:] = []

            wblist_Adam[:] = []
            walist_Adam[:] = []

        wa_SGD, wb_SGD = WaTraces_SGD[i], WbTraces_SGD[i]
        wblist_SGD.append(wa_SGD)
        walist_SGD.append(wb_SGD)
        point_SGD.set_data(wa_SGD, wb_SGD)
        line_SGD.set_data(wblist_SGD, walist_SGD)

        wa_SGD_MOMENTUM, wb_SGD_MOMENTUM  = WaTraces_SGD_MOMENTUM[i], WbTraces_SGD_MOMENTUM[i]
        wblist_SGD_MOMENTUM.append(wa_SGD_MOMENTUM)
        walist_SGD_MOMENTUM.append(wb_SGD_MOMENTUM)
        point_SGD_MOMENTUM.set_data(wa_SGD_MOMENTUM, wb_SGD_MOMENTUM)
        line_SGD_MOMENTUM.set_data(wblist_SGD_MOMENTUM, walist_SGD_MOMENTUM)

        wa_RMSprop, wb_RMSprop = WaTraces_RMSprop[i], WbTraces_RMSprop[i]
        wblist_RMSprop.append(wa_RMSprop)
        walist_RMSprop.append(wb_RMSprop)
        point_RMSprop.set_data(wa_RMSprop, wb_RMSprop)
        line_RMSprop.set_data(wblist_RMSprop, walist_RMSprop)

        wa_Adam, wb_Adam = WaTraces_Adam[i], WbTraces_Adam[i]
        wblist_Adam.append(wa_Adam)
        walist_Adam.append(wb_Adam)
        point_Adam.set_data(wa_Adam, wb_Adam)
        line_Adam.set_data(wblist_Adam, walist_Adam)

        update, epoch, minibatch = updateTraces[i], EpochTraces[i]+1, minibatchTraces[i]+1
        text_update.set_text('Epoch={:d}, minibatch={:d}, Updates={:d}'.format(epoch, minibatch, update))

        return point_SGD, line_SGD, point_SGD_MOMENTUM, line_SGD_MOMENTUM, point_RMSprop, line_RMSprop, point_Adam, line_Adam, text_update

    # call the animator.  blit=True means only re-draw the parts that have changed.
    intervalms = 10  # this means 10 ms per frame
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nframes, interval=intervalms, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be installed.
    anim.save('results/Part5_Fig3_Animation_of_loss_vs_TrainingUpdates.mp4', fps=30, bitrate=1800)

    plt.show()

    print('Done!')

if __name__ == '__main__':
    main()

