import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import math
from matplotlib import pyplot as plt
from Ellipse_Dataset import EllipseDataset


class OrbitRegressionNet(nn.Module):
    def __init__(self, xaxisoffset, initWa, initWb):
        super(OrbitRegressionNet, self).__init__()
        self.xaxisoffset = xaxisoffset
        self.Wa = nn.Parameter(initWa)
        self.Wb = nn.Parameter(initWb)

    def forward(self, x):
        y_pred_sqr = 1.0 - (x + self.xaxisoffset) ** 2 / self.Wa ** 2
        negativeloc = y_pred_sqr < 0  # record the non-negative y_pred_sqr elements, to be used later
        y_pred_sqr[negativeloc] = 0  # handle negative values caused by noise
        y_pred = torch.sqrt(y_pred_sqr) * self.Wb
        # return y_pred and negativeloc. negativeloc will be used to exclude those voxels from loss calculation.
        return y_pred, negativeloc


def main():
    flag_plot_final_result = True
    flag_write_log = True

    device = torch.device('cpu')
    torch.manual_seed(9999)
    # a and b are used to generate the training dataset
    a = 1.261845
    b = 1.234378
    c = math.sqrt(a * a - b * b)
    xaxisoffset = c  # offset of the center of the ellipse from the origin in this case
    nsamples = 512
    batch_size = 64
    epoch = 100

    # initWa = torch.rand([], device=device, requires_grad=True)
    # initWb = torch.rand([], device=device, requires_grad=True)
    # to initialize Wa and Wb with specific starting values
    initWa = torch.tensor(0.10, device=device, requires_grad=True)
    initWb = torch.tensor(1.8, device=device, requires_grad=True)
    thenet = OrbitRegressionNet(xaxisoffset, initWa, initWb)
    loss_fn = nn.MSELoss(reduction='mean')

    optim_algo = 'Adam'  # 'SGD', 'SGD_Momentum', 'RMSprop' or 'Adam', case sensitive
    if optim_algo == 'SGD':
        # ----- hyper-parameters for the SGD optimizer -------------------
        init_learning_rate = 0.01
        momentum = 0.9
        optimizer = optim.SGD(thenet.parameters(), lr=init_learning_rate, momentum=0.0)
    elif optim_algo == 'SGD_Momentum':
        # ----- hyper-parameters for the SGD_with_Momentum optimizer -----
        init_learning_rate = 0.01
        momentum = 0.9
        dampening = 0.0
        optimizer = optim.SGD(thenet.parameters(), lr=init_learning_rate, momentum=momentum, dampening=dampening)
    elif optim_algo == 'RMSprop':
        # ----- hyper-parameters for the RMSprop optimizer ---------------
        init_learning_rate = 0.02
        alpha = 0.99
        eps = 1e-08
        optimizer = optim.RMSprop(thenet.parameters(), lr=init_learning_rate, alpha=alpha, eps=eps)
    else:
        # ----- hyper-parameters for the Adam optimizer --------
        init_learning_rate = 0.12
        beta1, beta2 = 0.9, 0.999
        eps = 1e-08
        optimizer = optim.Adam(thenet.parameters(), lr=init_learning_rate, betas=(beta1, beta2), eps=eps)

    # instantiate the MultiplicativeLR schedule class, comment out one of the following two blocks
    # lambda_fn_0_8 = lambda epoch: 0.8 if (epoch >= 6 and epoch <= 24 and epoch % 2 == 0) else 1.0
    # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda_fn_0_8)
    # strLambda = 'lambda_fn_0_8'
    lambda0_5 = lambda epoch: 0.5 if (epoch >= 5 and epoch <= 20 and epoch % 5 == 0) else 1.0
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda0_5)
    strLambda = 'lambda_fn_0_5'

    if flag_write_log:
        logfilename = r'results/' + optim_algo + '_lr{}'.format(init_learning_rate)+'_Epoch{}'.format(epoch)+ '_Schl_{}'.format(strLambda) +'_results.log'
        foutput = open(logfilename, "w")
        foutput.write(optim_algo + 'result\n')
        logstr = 'nsamples={}, batch_size={}, epoch={}, lr={}, lr_gamma={}\ninitial Wa={}, Wb={}, lr_milestone={}\n' \
            .format(nsamples, batch_size, epoch, init_learning_rate, strLambda, initWa, initWb,
                    r'lmbda0_5 = lambda epoch: 0.5 if (epoch >= 5 and epoch <= 20 and epoch % 5 == 0) else 1.0')
        foutput.write(logstr)

    xy_dataset = EllipseDataset(nsamples, a, b, noise_scale=0.1)
    xy_dataloader = DataLoader(xy_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    updateCount = 0

    for t in range(epoch):

        for i_batch, sample_batched in enumerate(xy_dataloader):
            x, y = sample_batched['x'], sample_batched['y']

            thenet.train()

            # Step 1: Perform forward pass
            y_pred, negativeloc = thenet(x)

            # Step 2: Compute loss
            loss = loss_fn(y_pred[~negativeloc], y[~negativeloc])

            # Step 3: perform back-propagation and calculate the gradients of loss w.r.t. Wa and Wb
            optimizer.zero_grad()
            loss.backward()

            if flag_write_log:
                logstr = 'updates={}, Epoch={}, minibatch={}, loss={:.5f}, Wa={:.4f}, Wb={:.4f}'.format(
                    updateCount+1, t, i_batch, loss.item(), thenet.Wa.data.numpy(), thenet.Wb.data.numpy())
                logstr1 = ', unitvecWa={:.5f}, unitvecWb={:.5f}, learningrate={:.5f}, ' \
                         'dWa={:.5f}, dWb={:.5f}, VdWa={:.5f}, VdWb={:.5f}, sqrt_SdWa={:.5f}, sqrt_SdWb={:.5f}\n'.format(
                    0, 0, optimizer.param_groups[0]['lr'], thenet.Wa.grad.data.numpy(), thenet.Wb.grad.data.numpy(), 0, 0, 0, 0)
                foutput.write(logstr+logstr1)
                # if t % 10 == 0 and i_batch == 0:
                #     print(logstr)

            # Step 4: finally Update weights Wa and Wb using Adam algorithm.
            optimizer.step()
            # Step 4.1: and update the learning rate hyper-parameter based on the scheduler.
            scheduler.step()

            updateCount += 1

    # log the final results
    if flag_write_log:
        logstr = 'The ground truth is A={:.4f}, B={:.4f}\n'.format(a, b)
        logstr += 'PyTorch built-in AutoGradient+optimizer result: Final estimated Wa={:.4f}, Wb={:.4f}\n'.format(thenet.Wa, thenet.Wb)
        foutput.write(logstr)
        foutput.close()
        print(logstr)

    # plot the results obtained from the training
    if flag_plot_final_result:
        x = xy_dataset[:]['x']
        yfit = thenet.Wb * torch.sqrt(1.0 - (x+c)**2 / thenet.Wa**2)
        yfit[yfit!=yfit] = 0.0  # take care of the "Nan" at the end-points due to sqrt(negative_value_caused_by_noise)
        plt.plot(x, yfit.detach().numpy(), color="purple", linewidth=2.0)
        strEquation = r'$\frac{{{\left({x+' + '{:.3f}'.format(c) + r'}\right)}^2}}{{' + '{:.3f}'.\
            format(thenet.Wa) + r'^2}}+\frac{y^2}{' + '{:.3f}'.format(thenet.Wb) + r'^2}=1$'
        x0, y0 = x.detach().numpy()[nsamples * 2 // 3], yfit.detach().numpy()[nsamples * 2 // 3]
        plt.annotate(strEquation, xy=(x0, y0), xycoords='data',
                     xytext=(+0.75, 1.75), textcoords='data', fontsize=16,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.text(1.0, 1.5, 'Result of Adam', color='black', fontsize=12)
        plt.text(-1.9, 1.9,
                 'Epoch={}\n'.format(epoch) + 'Init Learning Rate={}\n'.format(init_learning_rate) + 'MultiplicativeLR w/ {}'.format(strLambda),
                 color='black', fontsize=12, ha='left', va='top')

        figfilename = r'results/' + optim_algo + '_lr{}'.format(init_learning_rate) + '_Epoch{}'.format(epoch) + '_Schl_{}'.format(strLambda) +  '.png'
        plt.savefig(figfilename)
        plt.show()

    print('Done!')

if __name__ == '__main__':
    main()

