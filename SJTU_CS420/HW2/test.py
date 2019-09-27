from FA import FA
from dataset import Dataset
import matplotlib.pyplot as plt

aic_n = []
bic_n = []
def testFA(data, n, aic=True):
    model = FA(n_components=n, Data=data)
    if aic:
        print('------------test FA with aic selection------------')
        model.aic_select()
        aic_n.append(model.show())
    else:
        print('------------test FA with bic selection------------')
        model.bic_select()
        bic_n.append(model.show())

def sample_size():
    '''
    sample size experiment
    :return: None
    '''
    samples = [30, 50, 100, 200, 300]

    for n in samples:
        data = Dataset(data_num=n)
        data.generate()
        testFA(data, 5)
        testFA(data, 5, False)

    plt.figure()
    plt.xlabel('Sample Size')
    plt.xticks(samples)
    plt.ylabel('Selected Dimensionality of y')
    plt.ylim((0, 5))

    point_size = 20
    plt.scatter(samples, bic_n, s=point_size)
    plt.scatter(samples, aic_n, s=point_size)

    ax1 = plt.plot(samples, bic_n, label='BIC', linestyle='--')
    ax2 = plt.plot(samples, aic_n, label='AIC', linestyle='--')

    plt.legend()
    plt.savefig('fig/sample_line')
    plt.show()

def m_dim():
    '''
    m_dimensionality experiment
    :return: None
    '''
    dim_size = [2, 3, 4, 5, 6]
    for n in dim_size:
        data = Dataset(m_dim=n, data_num=500, data_dim=10, var=0.1, mu=0)
        data.generate()
        testFA(data, 10)
        testFA(data, 10, False)

    plt.figure()
    plt.xlabel('Dimensionality of y')
    plt.xticks(dim_size)
    plt.ylabel('Selected Dimensionality of y')
    plt.ylim((0, 10))

    point_size = 20
    plt.scatter(dim_size, bic_n, s=point_size)
    plt.scatter(dim_size, aic_n, s=point_size)

    ax1 = plt.plot(dim_size, bic_n, label='BIC', linestyle='--')
    ax2 = plt.plot(dim_size, aic_n, label='AIC', linestyle='--')

    plt.legend()
    plt.savefig('fig/dim2_line')
    plt.show()

def n_dim():
    '''
    m_dimensionality experiment
    :return: None
    '''
    dim_size = [8, 9, 10, 11, 12, 13]
    for n in dim_size:
        data = Dataset(m_dim=3, data_num=2000, data_dim=n, var=0.1, mu=0)
        data.generate()
        testFA(data, 5)
        testFA(data, 5, False)

    plt.figure()
    plt.xlabel('Dimensionality of data')
    plt.xticks(dim_size)
    plt.ylabel('Selected Dimensionality of y')
    plt.ylim((0, 10))

    point_size = 20
    plt.scatter(dim_size, bic_n, s=point_size)
    plt.scatter(dim_size, aic_n, s=point_size)

    ax1 = plt.plot(dim_size, bic_n, label='BIC', linestyle='--')
    ax2 = plt.plot(dim_size, aic_n, label='AIC', linestyle='--')

    plt.legend()
    plt.savefig('fig/n_dim2_line')
    plt.show()

def noise():
    '''
    noise experiment
    :return: None
    '''
    noise = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    for n in noise:
        data = Dataset(m_dim=3, data_num=2000, data_dim=10, var=n, mu=0)
        data.generate()
        testFA(data, 5)
        testFA(data, 5, False)

    plt.figure()
    plt.xlabel('Noise Level')
    plt.xticks(noise)
    plt.ylabel('Selected Dimensionality of y')
    plt.ylim((0, 10))

    point_size = 20
    plt.scatter(noise, bic_n, s=point_size)
    plt.scatter(noise, aic_n, s=point_size)

    ax1 = plt.plot(noise, bic_n, label='BIC', linestyle='--')
    ax2 = plt.plot(noise, aic_n, label='AIC', linestyle='--')

    plt.legend()
    plt.savefig('fig/noise2_line')
    plt.show()

if __name__ == '__main__':
    #sample_size()
    #m_dim()
    n_dim()
    #noise()