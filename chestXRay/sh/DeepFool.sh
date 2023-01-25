weps1=0.5
python src/DeepFool.py --AK=white --eps=${weps1}  --output_path=output/DeepFool/white > output/DeepFool/white/eps${weps1}.txt

weps2=0.25
python src/DeepFool.py --AK=white --eps=${weps2}  --output_path=output/DeepFool/white > output/DeepFool/white/eps${weps2}.txt

peps1=0.5
python src/DeepFool.py --AK=proxy --eps=${peps1}  --output_path=output/DeepFool/proxy > output/DeepFool/proxy/eps${peps1}.txt

peps2=0.25
python src/DeepFool.py --AK=proxy --eps=${peps2}  --output_path=output/DeepFool/proxy > output/DeepFool/proxy/eps${peps2}.txt

