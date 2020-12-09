import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy


def truncate(li):
	#print([len(x) for x in li])
	N = numpy.min([len(x) for x in li])
	return [l[:N] for l in li]


def smooth(li):
	window = 2
	y = li
	y_smooth = [numpy.mean(y[max(x - window, 0):x + window]) for x in range(len(y))]
	return y_smooth


def compute_fina_mean_and_std(li):
	li_last = [l[-1] for l in li]
	print(numpy.mean(li_last))
	print(numpy.std(li_last) / 20)

problems_name = [
    'Pendulum',
    'LunarLander',
    'Bipedal',
    'Hopper',
    'Cheetah',
    'Ant',
    'InvertedDoublePendulum',
    'InvertedPendulum',
    'Reacher'
]
ylim_down = [-1500, -350, -100, -500, -500, -500, 0, 0, -80]
ylim_up = [-100, 235, 300, 3000, 8000, 3000, 9350, 1000, -4]
#y_ticks = [[-1000,-150],[-200,220],[0,250],[0,2500],[0,7500],[0,3000],[0,9000],[0,1000],[-50,-4]]
#setting_li=[0]
#setting_li=[0]+list(range(900,910))
#setting_li=[0]
labels = [
    '100 updates',
    '200 updates',
    '200 updates slow target',
    '200 updates slowest target',
    '200 updates slowest! target',
    '200 updates less target updates',
    '200 updates less target updates'
]
colors = ['blue', 'black', 'brown', 'orange', 'green', 'yellow', 'black', 'purple']
#for problem in [4]:
for problem in [1,2]:
	plt.subplot(4, 2, problem + 1)
	print(problems_name[problem])
	for setting in range(5,10):
	#for setting in range(4):
		hyper_parameter_name = str(problem) + str(setting)
		acceptable_len = 00
		li = []
		for seed_num in range(20):
			try:
				temp = numpy.loadtxt("rbf_results/" + str(hyper_parameter_name) +"/" + str(seed_num) + ".txt")
				#plt.plot(smooth(temp),lw=1,color=colors[setting%len(colors)])
				if len(temp) > acceptable_len:
					li.append(temp)
					
					print(hyper_parameter_name,seed_num,numpy.mean(temp[-1:]),len(temp))	
			except:
				#print("problem")
				pass
		#print([len(x) for x in li])
		li = truncate(li)
		print(hyper_parameter_name,len(li[0]),
		      numpy.mean(numpy.mean(li, axis=0)[-1:]),numpy.mean(li),len(li))
		plt.plot(smooth(numpy.mean(li, axis=0)), label=setting, lw=5, color=colors[setting%len(colors)])
		plt.tick_params( labelright=True)
		#plt.ylim([ylim_down[problem],ylim_up[problem]])
		#plt.yticks([ylim_down[problem],ylim_up[problem]])
	plt.title(problems_name[problem])
	#plt.legend()
plt.subplots_adjust(wspace=0.5, hspace=1)
plt.show()
