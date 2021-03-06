import torch
from torch.utils.data import DataLoader
import math
from matplotlib import pyplot as plt
from Ellipse_Dataset import EllipseDataset
from torch import optim


def main():
    flag_manual_implement = True  # True: using our custom implementation; False: using Torch built-in
    flag_plot_final_result = True
    flag_log = True

    device = torch.device('cpu')
    torch.manual_seed(9999)
    a = 1.261845
    b = 1.234378
    c = math.sqrt(a * a - b * b)
    nsamples = 512
    batch_size = 64
    epoch = 160
    learning_rate = 0.01 # 0.02
    lr_milestones = [100]
    lr_gamma = 1.0  # 1.0 means no LR scheduler

    xy_dataset = EllipseDataset(nsamples, a, b, noise_scale=0.1)
    xy_dataloader = DataLoader(xy_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Wa = torch.rand([], device=device, requires_grad=True)
    # Wb = torch.rand([], device=device, requires_grad=True)
    Wa = torch.tensor(0.10, device=device, requires_grad=True)
    Wb = torch.tensor(1.8, device=device, requires_grad=True)

    if flag_log:
        if flag_manual_implement:
            logfilename = 'results/SGD_custom_implement_LR{}'.format(learning_rate) + '_results.log'
            foutput = open(logfilename, "w")
            foutput.write('SGD optimization using custom implementation' + '\n')
        else:
            logfilename = 'results/SGD_custom_implement_LR{}'.format(learning_rate) + '_results.log'
            foutput = open(logfilename, "w")
            foutput.write('SGD optimization using torch built-in' + '\n')
        logstr = 'nsamples={}, batch_size={}, epoch={}, lr={}\ninitial Wa={}, Wb={}\n'\
            .format(nsamples, batch_size, epoch, learning_rate, Wa, Wb)
        foutput.write(logstr)
        print(logstr)

    if flag_manual_implement:
        optimizer = None
    else:
        optimizer = optim.SGD([Wa, Wb], lr=learning_rate, momentum=0.0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)

    updates = 0
    for t in range(epoch):
        for i_batch, sample_batched in enumerate(xy_dataloader):
            x, y = sample_batched['x'], sample_batched['y']

            # Step 1: Perform forward pass
            y_pred_sqr = 1.0 - (x + c) ** 2 / Wa ** 2
            negativeloc = y_pred_sqr < 0  # record the non-negative y_pred_sqr elements, to be used later
            y_pred_sqr[negativeloc] = 0  # handle negative values caused by noise
            y_pred = torch.sqrt(y_pred_sqr) * Wb

            # Step 2: Compute loss
            loss = ((y_pred - y).pow(2))[~negativeloc].mean()

            if flag_log:
                logstr = 'updates={}, Epoch={}, minibatch={}, loss={:.5f}, Wa={:.4f}, Wb={:.4f}'.format(
                    updates+1, t, i_batch, loss.item(), Wa.data.numpy(), Wb.data.numpy())
                foutput.write(logstr)
                # if t % 10 == 0 and i_batch == 0:
                #     print(logstr)

            if flag_manual_implement:  # do the job manually
                if updates in lr_milestones:
                    learning_rate *= lr_gamma

                # Step 3: perform back-propagation and calculate the gradients of loss w.r.t. Wa and Wb
                dWa_via_yi = (2.0 * (y_pred - y) * ((x + c) ** 2) * (Wb ** 2) / (Wa ** 3) / y_pred)
                dWa = dWa_via_yi[~negativeloc].mean()  # sum()

                dWb_via_yi = ((2.0 * (y_pred - y) * y_pred / Wb))
                dWb = dWb_via_yi[~negativeloc].mean()  # .sum()

                # Step 4: Update weights using Gradient Descent algorithm.
                with torch.no_grad():
                    Wa -= learning_rate * dWa
                    Wb -= learning_rate * dWb

                    if flag_log:
                        mag_dWadWb = math.sqrt(dWa ** 2 + dWb ** 2)
                        unitvec_a = dWa / mag_dWadWb
                        unitvec_b = dWb / mag_dWadWb
                        actual_stepsize = learning_rate * mag_dWadWb
                        logstr = ', unitvecWa={:.5f}, unitvecWb={:.5f}, eff_stepsize={:.5f}, ' \
                                 'dWa={:.5f}, dWb={:.5f}, VdWa={:.5f}, VdWb={:.5f}, sqrt_SdWa={:.5f}, sqrt_SdWb={:.5f}\n'.format(
                            unitvec_a, unitvec_b, actual_stepsize, dWa, dWb, 0, 0, 0, 0)
                        foutput.write(logstr)

            else:  # do the same job using Torch built-in autograd and optim
                optimizer.zero_grad()
                # Step 3: perform back-propagation and calculate the gradients of loss w.r.t. Wa and Wb
                loss.backward()
                # Step 4: Update weights using Adam algorithm.
                optimizer.step()
                scheduler.step()

            updates += 1

    # log the final results
    if flag_log:
        logstr = 'The ground truth is A={:.4f}, B={:.4f}\n'.format(a, b)
        if flag_manual_implement:
            logstr += 'Manually implemented gradient+optimizer result: Final estimated Wa={:.4f}, Wb={:.4f}\n'.format(Wa, Wb)
        else:
            logstr += 'PyTorch built-in AutoGradient+optimizer result: Final estimated Wa={:.4f}, Wb={:.4f}\n'.format(Wa, Wb)
        foutput.write(logstr)
        foutput.close()
        print(logstr)

    # plot the results obtained from the training
    if flag_plot_final_result:
        x = xy_dataset[:]['x']
        yfit = Wb * torch.sqrt(1.0-(x+c)**2/Wa**2)
        yfit[yfit!=yfit] = 0.0  # take care of the "Nan" at the end-points due to sqrt(negative_value_caused_by_noise)
        plt.plot(x, yfit.detach().numpy(), color="purple", linewidth=2.0)
        strEquation = r'$\frac{{{\left({x+' + '{:.3f}'.format(c) + r'}\right)}^2}}{{' + '{:.3f}'.format(Wa) + r'^2}}+\frac{y^2}{' + '{:.3f}'.format(Wb) + r'^2}=1$'
        x0, y0 = x.detach().numpy()[nsamples * 2 // 3], yfit.detach().numpy()[nsamples * 2 // 3]
        plt.annotate(strEquation, xy=(x0, y0), xycoords='data',
                     xytext=(+0.75, 1.75), textcoords='data', fontsize=16,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.text(1.0, 1.5, 'Result of SGD', color='black', fontsize=12)
        plt.show()

    print('Done!')


if __name__ == '__main__':
    main()

