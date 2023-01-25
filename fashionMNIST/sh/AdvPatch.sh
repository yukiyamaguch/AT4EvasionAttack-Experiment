scale1=0.15
scale2=0.2
scale3=0.3

# normal, proxy0.5
python src/AdvPatch.py --AK=proxy0.5 --scale=${scale1} --output_path=output/AdvPatch/proxy0.5 > output/AdvPatch/proxy0.5/scale${scale1}.txt
python src/AdvPatch.py --AK=proxy0.5 --scale=${scale2} --output_path=output/AdvPatch/proxy0.5 > output/AdvPatch/proxy0.5/scale${scale2}.txt
python src/AdvPatch.py --AK=proxy0.5 --scale=${scale3} --output_path=output/AdvPatch/proxy0.5 > output/AdvPatch/proxy0.5/scale${scale3}.txt

# adv_train, proxy
python src/AdvPatch.py --AK=robust_white --scale=${scale1} --output_path=output/AdvPatch/robust_white > output/AdvPatch/robust_white/scale${scale1}.txt
python src/AdvPatch.py --AK=robust_white --scale=${scale2} --output_path=output/AdvPatch/robust_white > output/AdvPatch/robust_white/scale${scale2}.txt
python src/AdvPatch.py --AK=robust_white --scale=${scale3} --output_path=output/AdvPatch/robust_white > output/AdvPatch/robust_white/scale${scale3}.txt

python src/AdvPatch.py --AK=robust_proxy --scale=${scale1} --output_path=output/AdvPatch/robust_proxy > output/AdvPatch/robust_proxy/scale${scale1}.txt
python src/AdvPatch.py --AK=robust_proxy --scale=${scale2} --output_path=output/AdvPatch/robust_proxy > output/AdvPatch/robust_proxy/scale${scale2}.txt
python src/AdvPatch.py --AK=robust_proxy --scale=${scale3} --output_path=output/AdvPatch/robust_proxy > output/AdvPatch/robust_proxy/scale${scale3}.txt

# adv_train, proxy0.5
python src/AdvPatch.py --AK=robust_proxy0.5 --scale=${scale1} --output_path=output/AdvPatch/robust_proxy0.5 > output/AdvPatch/robust_proxy0.5/scale${scale1}.txt
python src/AdvPatch.py --AK=robust_proxy0.5 --scale=${scale2} --output_path=output/AdvPatch/robust_proxy0.5 > output/AdvPatch/robust_proxy0.5/scale${scale2}.txt
python src/AdvPatch.py --AK=robust_proxy0.5 --scale=${scale3} --output_path=output/AdvPatch/robust_proxy0.5 > output/AdvPatch/robust_proxy0.5/scale${scale3}.txt
