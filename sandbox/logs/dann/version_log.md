v2 
- note
  - replicate `https://tl.thuml.ai/get_started/quickstart.html`
  - `CUDA_VISIBLE_DEVICES=0 python dann.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log logs/dann/Office31_A2W`
  - note that there's no separate test set for the target domain
  - train:
    - source train =  `amazon` (x+y)
    - target train = `webcam` (x)
  - validation: (choose the best model)
    - source val =  `webcam` (x+y)    
  - test:
    - target test =  `webcam`  (x+y)      
- metric
  - test acc (target domain): * Acc@1 89.686

v3
- note
  - dataset: `office31_v2`
  - evaluate the model both on the test sets of the source and the target domain 
  - train:
    - source train =  `amazon_train` (x+y)
    - target train = `webcam_train` (x)
  - validation: (choose the best model)
    - source val =  `amazon_val` (x+y)    
  - test:
    - source test =  `amazon_test` (x+y)
    - target test =  `webcam_test`  (x+y)    
  - metric
  Top-1 accuracy on source_train_loader: 97.65625
  Top-1 accuracy on target_train_loader: 81.02678571428571
  Top-1 accuracy on source_val_loader: 86.6785078709333
  Top-1 accuracy on target_val_loader: 86.16352186862778
  Top-1 accuracy on source_test_loader: 87.41134751773049
  Top-1 accuracy on target_test_loader: 84.90566100114546   

v4
- note
  - dataset: `office31_v2`
  - based on v3 but turn off the adversarial loss 
    - only train the model using the supervised loss in the source domain
    - this can serve as a baseline model
  - train:
    - source train =  `amazon_train` (x+y)
    - target train = `webcam_train` (x)
  - validation: (choose the best model)
    - source val =  `amazon_val` (x+y)    
  - test:
    - source test =  `amazon_test` (x+y)
    - target test =  `webcam_test`  (x+y)    
  - metric
  Top-1 accuracy on source_train_loader: 98.49759615384616
  Top-1 accuracy on target_train_loader: 49.107142857142854
  Top-1 accuracy on source_val_loader: 87.38898753950481
  Top-1 accuracy on target_val_loader: 57.86163558000289
  Top-1 accuracy on source_test_loader: 88.29787234042553
  Top-1 accuracy on target_test_loader: 55.97484283927102  
  - summary
    - turning off the adversarial loss DOES harm the cross-domain accuracy

v5
- note
  - same as v3 except that using wandb to record the process
  - dataset: `office31_v2`
  - evaluate the model both on the test sets of the source and the target domain 
  - train:
    - source train =  `amazon_train` (x+y)
    - target train = `webcam_train` (x)
  - validation: (choose the best model)
    - source val =  `amazon_val` (x+y)    
  - test:
    - source test =  `amazon_test` (x+y)
    - target test =  `webcam_test`  (x+y)    
  - metric
v6
- note
  - same as v4 except that using wandb to record the process 
  - dataset: `office31_v2`
  - based on v3 but turn off the adversarial loss 
    - only train the model using the supervised loss in the source domain
    - this can serve as a baseline model
  - train:
    - source train =  `amazon_train` (x+y)
    - target train = `webcam_train` (x)
  - validation: (choose the best model)
    - source val =  `amazon_val` (x+y)    
  - test:
    - source test =  `amazon_test` (x+y)
    - target test =  `webcam_test`  (x+y)     