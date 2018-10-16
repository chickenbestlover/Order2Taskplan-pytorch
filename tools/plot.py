from matplotlib import pyplot as plt
import pickle
'''
def savePlot(args,ppl_trains,ppl_tests,EMs,F1s):

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    num_epoch =len(ppl_trains)
    plt.plot(range(num_epoch), ppl_trains,'r',label='train')
    plt.plot(range(num_epoch), ppl_tests,'b',label='test')
    plt.xlim([0,50])
    plt.ylim([1,2])
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(num_epoch), EMs,'r', label='Exact Match')
    plt.plot(range(num_epoch), F1s,'b', label='F1')
    plt.xlim([0,50])

    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    #pickle.dump(fig, open(args.plot_file+'.pickle', 'wb'))
    plt.savefig(args.plot_file)
    plt.close()
'''

def savePlot(args,ppl_trains,ppl_tests,EMs,F1s):

    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121)

    ax1.plot(range(len(ppl_trains)), ppl_trains,'r',label='train')
    ax1.plot(range(len(ppl_tests)), ppl_tests,'b',label='test')
    ax1.set_xlim([-50,250])
    ax1.set_ylim([1,4])
    ax1.grid()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Perplexity')
    ax1.legend()


    ax2 = fig.add_subplot(122)
    ax2.plot(range(len(EMs)), EMs,'r', label='Exact Match')
    ax2.plot(range(len(F1s)), F1s,'b', label='F1')
    ax2.set_xlim([-50,250])
    ax2.grid()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    file= open(args.plot_file[:-3]+'pickle', 'wb')
    pickle.dump(fig, file)
    file.close()
    fig.savefig(args.plot_file)
    plt.close()



def savePlot_hall1(args,loss_trains,ppl_tests,EMs,F1s):

    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(221)
    num_epoch =len(loss_trains)
    ax1.plot(range(num_epoch),loss_trains,'r',label='train')
    ax1.set_xlim([-50,150])
    ax1.grid()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()

    ax2 = fig.add_subplot(222)
    for color, state in zip(['r--','g--','b--'], ['normal','none','hall']):
        ax2.plot(range(num_epoch),ppl_tests[state],color,label=state)
#    for color, state in zip(['r','g','b'], ['normal+behavior','none+behavior','hall+behavior']):
#        ax2.plot(range(num_epoch),ppl_tests[state],color,label=state)

    ax2.set_xlim([-50,150])
    ax2.grid()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.legend()


    ax3 = fig.add_subplot(223)
    for color, state in zip(['r--','g--','b--'], ['normal','none','hall']):
        ax3.plot(range(num_epoch),EMs[state],color,label=state)
 #   for color, state in zip(['r','g','b'], ['normal+behavior','none+behavior','hall+behavior']):
  #      ax3.plot(range(num_epoch),EMs[state],color,label=state)

    ax3.set_xlim([-50,150])
    ax3.grid()
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Exact Match')
    ax3.legend()

    ax4 = fig.add_subplot(224)
    for color, state in zip(['r--','g--','b--'], ['normal','none','hall']):
        ax4.plot(range(num_epoch),F1s[state],color,label=state)
   # for color, state in zip(['r','g','b'], ['normal+behavior','none+behavior','hall+behavior']):
    #    ax4.plot(range(num_epoch),F1s[state],color,label=state)

    ax4.set_xlim([-50,150])
    ax4.grid()
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 score')
    ax4.legend()
    file= open(args.plot_file[:-4]+'1.pickle', 'wb')
    pickle.dump(fig, file)
    file.close()
    plt.savefig(args.plot_file[:-4]+'1.pdf')
    plt.close()

def savePlot_hall2(args,loss_trains,ppl_tests,EMs,F1s):

    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(221)
    num_epoch =len(loss_trains)
    ax1.plot(range(num_epoch),loss_trains,'r',label='train')
    ax1.set_xlim([-50,150])
    ax1.grid()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()

    ax2 = fig.add_subplot(222)
#    for color, state in zip(['r--','g--','b--'], ['normal','none','hall']):
 #       ax2.plot(range(num_epoch),ppl_tests[state],color,label=state)
    for color, state in zip(['r','g','b'], ['normal+behavior','none+behavior','hall+behavior']):
        ax2.plot(range(num_epoch),ppl_tests[state],color,label=state)

    ax2.set_xlim([-50,150])
    ax2.grid()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.legend()


    ax3 = fig.add_subplot(223)
  #  for color, state in zip(['r--','g--','b--'], ['normal','none','hall']):
   #     ax3.plot(range(num_epoch),EMs[state],color,label=state)
    for color, state in zip(['r','g','b'], ['normal+behavior','none+behavior','hall+behavior']):
        ax3.plot(range(num_epoch),EMs[state],color,label=state)

    ax3.set_xlim([-50,150])
    ax3.grid()
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Exact Match')
    ax3.legend()

    ax4 = fig.add_subplot(224)
    #for color, state in zip(['r--','g--','b--'], ['normal','none','hall']):
     #   ax4.plot(range(num_epoch),F1s[state],color,label=state)
    for color, state in zip(['r','g','b'], ['normal+behavior','none+behavior','hall+behavior']):
        ax4.plot(range(num_epoch),F1s[state],color,label=state)

    ax4.set_xlim([-50,150])
    ax4.grid()
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 score')
    ax4.legend()
    file= open(args.plot_file[:-4]+'2.pickle', 'wb')
    pickle.dump(fig, file)
    file.close()
    plt.savefig(args.plot_file[:-4]+'2.pdf')
    plt.close()





def savePlot_hall(args,loss_trains,ppl_tests,EMs,F1s):

    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(221)
    num_epoch =len(loss_trains)
    ax1.plot(range(num_epoch),loss_trains,'r',label='train')
    ax1.set_xlim([0,75])
    ax1.grid()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()

    ax2 = fig.add_subplot(222)
    for color, state in zip(['r--','g--','b--'], ['normal','none','hall']):
        ax2.plot(range(num_epoch),ppl_tests[state],color,label=state)
    for color, state in zip(['r','g','b'], ['normal+behavior','none+behavior','hall+behavior']):
        ax2.plot(range(num_epoch),ppl_tests[state],color,label=state)

    ax2.set_xlim([0,75])
    ax2.grid()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.legend()


    ax3 = fig.add_subplot(223)
    for color, state in zip(['r--','g--','b--'], ['normal','none','hall']):
        ax3.plot(range(num_epoch),EMs[state],color,label=state)
    for color, state in zip(['r','g','b'], ['normal+behavior','none+behavior','hall+behavior']):
        ax3.plot(range(num_epoch),EMs[state],color,label=state)

    ax3.set_xlim([0,75])
    ax3.grid()
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Exact Match')
    ax3.legend()

    ax4 = fig.add_subplot(224)
    for color, state in zip(['r--','g--','b--'], ['normal','none','hall']):
        ax4.plot(range(num_epoch),F1s[state],color,label=state)
    for color, state in zip(['r','g','b'], ['normal+behavior','none+behavior','hall+behavior']):
        ax4.plot(range(num_epoch),F1s[state],color,label=state)

    ax4.set_xlim([0,75])
    ax4.grid()
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 score')
    ax4.legend()
    file= open(args.plot_file[:-3]+'pickle', 'wb')
    pickle.dump(fig, file)
    file.close()
    plt.savefig(args.plot_file)
    plt.close()
