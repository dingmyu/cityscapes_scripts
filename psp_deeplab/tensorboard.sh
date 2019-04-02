rm exp/drivable/psp50_dist_16_713_fine/model/events.out.tfevents*                               
rm exp/drivable/psp50_dist_16_713_fine/model/*.log
tensorboard --logdir="exp/drivable/psp50_dist_16_713_fine/model" --port 9826
