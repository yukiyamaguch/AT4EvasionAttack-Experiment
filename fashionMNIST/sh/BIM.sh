eps1=0.05
eps2=0.075

# normal, proxy0.5
python src/BIM.py --AK=proxy0.5 --eps=${eps1}  --output_path=output/BIM/proxy0.5 > output/BIM/proxy0.5/eps${eps1}.txt
python src/BIM.py --AK=proxy0.5 --eps=${eps2}  --output_path=output/BIM/proxy0.5 > output/BIM/proxy0.5/eps${eps2}.txt

# adv_train, proxy
python src/BIM.py --AK=robust_white --eps=${eps1}  --output_path=output/BIM/robust_white > output/BIM/robust_white/eps${eps1}.txt
python src/BIM.py --AK=robust_white --eps=${eps2}  --output_path=output/BIM/robust_white > output/BIM/robust_white/eps${eps2}.txt

python src/BIM.py --AK=robust_proxy --eps=${eps1}  --output_path=output/BIM/robust_proxy > output/BIM/robust_proxy/eps${eps1}.txt
python src/BIM.py --AK=robust_proxy --eps=${eps2}  --output_path=output/BIM/robust_proxy > output/BIM/robust_proxy/eps${eps2}.txt

# adv_train, proxy0.5
python src/BIM.py --AK=robust_proxy0.5 --eps=${eps1}  --output_path=output/BIM/robust_proxy0.5 > output/BIM/robust_proxy0.5/eps${eps1}.txt
python src/BIM.py --AK=robust_proxy0.5 --eps=${eps2}  --output_path=output/BIM/robust_proxy0.5 > output/BIM/robust_proxy0.5/eps${eps2}.txt

