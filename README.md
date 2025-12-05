# Formal Verification for Deep Neural Network

### The vision experiments (ResNet with MNIST) 
```
cd formal_veri/auto_LiRPA/auto_LiRPA/examples/vision
# pretrain models in /models dir
# run training
python simple_training.py
# run evaluation
python simple_verification.py
```



### The NLP experiments (transformer with sst) 
```
cd formal_veri/auto_LiRPA/auto_LiRPA/examples/language/
# download sst dataset via sst.py
# pretrain models in /models dir
# run training
python train.py --train --device cpu \
    --model transformer --num_epochs 10 \
    --dir trained_model
# run eval
```
