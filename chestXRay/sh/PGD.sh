weps1=0.05
python src/PGD.py --AK=white --eps=${weps1}  --output_path=output/PGD/white > output/PGD/white/eps${weps1}.txt

weps2=0.075
python src/PGD.py --AK=white --eps=${weps2}  --output_path=output/PGD/white > output/PGD/white/eps${weps2}.txt

peps1=0.05
python src/PGD.py --AK=proxy --eps=${peps1}  --output_path=output/PGD/proxy > output/PGD/proxy/eps${peps1}.txt

peps2=0.075
python src/PGD.py --AK=proxy --eps=${peps2}  --output_path=output/PGD/proxy > output/PGD/proxy/eps${peps2}.txt

