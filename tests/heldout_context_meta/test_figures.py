from experiments.heldout_context_meta.figures import cases,PRIMARY

def test_no_selected_checkpoint_or_missing_baseline():
 assert cases(960)==[('Historical',0),('Seen',960),('Transfer',960)]
 assert cases(240)==[('Historical',0),('Seen',240),('Transfer',240)]
 assert PRIMARY==['figure_1','figure_2','figure_3','figure_4','figure_5']
