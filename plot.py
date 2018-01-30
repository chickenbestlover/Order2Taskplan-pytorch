from matplotlib import pyplot as plt

def savePlot(ppl_trains,ppl_tests,EMs,F1s):

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    num_epoch =len(ppl_trains)
    plt.plot(range(num_epoch),ppl_trains,'r',label='train')
    plt.plot(range(num_epoch),ppl_tests,'b',label='test')
    plt.xlim([-50,400])
    plt.ylim([1,4])
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(num_epoch), EMs,'r', label='Exact Match')
    plt.plot(range(num_epoch), F1s,'b', label='F1')
    plt.xlim([-50,400])

    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('result/scores_.pdf')

def savePlot_hall(loss_trains,ppl_tests,EMs,F1s):

    plt.figure(figsize=(18,6))
    plt.subplot(2,2,1)
    num_epoch =len(loss_trains)
    plt.plot(range(num_epoch),loss_trains,'r',label='train')
    plt.xlim([-50,400])
    #plt.ylim([1,4])
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()


    plt.subplot(2,2,2)
    for color, state in zip(['r','g','b'], ['normal','none','hall']):
        plt.plot(range(num_epoch),ppl_tests[state],color,label=state)

    plt.xlim([-50,400])
    #plt.ylim([1,4])
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()


    plt.subplot(2,2,3)
    for color, state in zip(['r','g','b'], ['normal','none','hall']):
        plt.plot(range(num_epoch),EMs[state],color,label=state)

    plt.xlim([-50,400])
    #plt.ylim([1,4])
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Exact Match')
    plt.legend()

    plt.subplot(2,2,4)
    for color, state in zip(['r','g','b'], ['normal','none','hall']):
        plt.plot(range(num_epoch),F1s[state],color,label=state)

    plt.xlim([-50,400])
    #plt.ylim([1,4])
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.legend()

    plt.savefig('result/scores_.pdf')

