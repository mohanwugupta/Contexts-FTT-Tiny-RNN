from experiments.meta_learning_scaling.figures import log_dose_axis,PRIMARY

def test_positive_doses_logarithmic_with_distinct_zero():
 import matplotlib.pyplot as plt
 f,ax=plt.subplots();log_dose_axis(ax,[0,40,240,960,1920]);assert ax.get_xscale()=='symlog';assert [t.get_text() for t in ax.get_xticklabels()]==['0','40','240','960','1920'];plt.close(f)
 assert PRIMARY==['figure_1','figure_2','figure_3','figure_4','figure_5']
