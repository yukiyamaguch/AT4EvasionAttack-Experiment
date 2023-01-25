weps1=0.025
python src/FGSM.py --AK=white --eps=${weps1}  --output_path=output/FGSM/white > output/FGSM/white/eps${weps1}.txt

weps2=0.05
python src/FGSM.py --AK=white --eps=${weps2}  --output_path=output/FGSM/white > output/FGSM/white/eps${weps2}.txt

peps1=0.025
python src/FGSM.py --AK=proxy --eps=${peps1}  --output_path=output/FGSM/proxy > output/FGSM/proxy/eps${peps1}.txt

peps2=0.05
python src/FGSM.py --AK=proxy --eps=${peps2}  --output_path=output/FGSM/proxy > output/FGSM/proxy/eps${peps2}.txt
