# All images for param = 0.3
python inference_img.py --img test_imgs/a1.png test_imgs/a3.png --exp=1 --model=final_models/param=0.3
mv output/img1.png "output/all_model_outs/a1_param=0.3.png"
python inference_img.py --img test_imgs/d1.png test_imgs/d3.png --exp=1 --model=final_models/param=0.3
mv output/img1.png "output/all_model_outs/d1_param=0.3.png"
python inference_img.py --img test_imgs/f1.png test_imgs/f3.png --exp=1 --model=final_models/param=0.3
mv output/img1.png "output/all_model_outs/f1_param=0.3.png"
python inference_img.py --img test_imgs/g1.png test_imgs/g3.png --exp=1 --model=final_models/param=0.3
mv output/img1.png "output/all_model_outs/g1_param=0.3.png"

# All images for param = 3
python inference_img.py --img test_imgs/a1.png test_imgs/a3.png --exp=1 --model=final_models/param=3
mv output/img1.png "output/all_model_outs/a1_param=3.png"
python inference_img.py --img test_imgs/d1.png test_imgs/d3.png --exp=1 --model=final_models/param=3
mv output/img1.png "output/all_model_outs/d1_param=3.png"
python inference_img.py --img test_imgs/f1.png test_imgs/f3.png --exp=1 --model=final_models/param=3
mv output/img1.png "output/all_model_outs/f1_param=3.png"
python inference_img.py --img test_imgs/g1.png test_imgs/g3.png --exp=1 --model=final_models/param=3
mv output/img1.png "output/all_model_outs/g1_param=3.png"

# All images for param = 30
python inference_img.py --img test_imgs/a1.png test_imgs/a3.png --exp=1 --model=final_models/param=30
mv output/img1.png "output/all_model_outs/a1_param=30.png"
python inference_img.py --img test_imgs/d1.png test_imgs/d3.png --exp=1 --model=final_models/param=30
mv output/img1.png "output/all_model_outs/d1_param=30.png"
python inference_img.py --img test_imgs/f1.png test_imgs/f3.png --exp=1 --model=final_models/param=30
mv output/img1.png "output/all_model_outs/f1_param=30.png"
python inference_img.py --img test_imgs/g1.png test_imgs/g3.png --exp=1 --model=final_models/param=30
mv output/img1.png "output/all_model_outs/g1_param=30.png"

# All images for param = 300
python inference_img.py --img test_imgs/a1.png test_imgs/a3.png --exp=1 --model=final_models/param=300
mv output/img1.png "output/all_model_outs/a1_param=300.png"
python inference_img.py --img test_imgs/d1.png test_imgs/d3.png --exp=1 --model=final_models/param=300
mv output/img1.png "output/all_model_outs/d1_param=300.png"
python inference_img.py --img test_imgs/f1.png test_imgs/f3.png --exp=1 --model=final_models/param=300
mv output/img1.png "output/all_model_outs/f1_param=300.png"
python inference_img.py --img test_imgs/g1.png test_imgs/g3.png --exp=1 --model=final_models/param=300
mv output/img1.png "output/all_model_outs/g1_param=300.png"

# All images for l2 loss
python inference_img.py --img test_imgs/a1.png test_imgs/a3.png --exp=1 --model=final_models/l2_loss
mv output/img1.png "output/all_model_outs/a1_l2_loss.png"
python inference_img.py --img test_imgs/d1.png test_imgs/d3.png --exp=1 --model=final_models/l2_loss
mv output/img1.png "output/all_model_outs/d1_l2_loss.png"
python inference_img.py --img test_imgs/f1.png test_imgs/f3.png --exp=1 --model=final_models/l2_loss
mv output/img1.png "output/all_model_outs/f1_l2_loss.png"
python inference_img.py --img test_imgs/g1.png test_imgs/g3.png --exp=1 --model=final_models/l2_loss
mv output/img1.png "output/all_model_outs/g1_l2_loss.png"
