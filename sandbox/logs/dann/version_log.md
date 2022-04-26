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
  - test acc: * Acc@1 89.686

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