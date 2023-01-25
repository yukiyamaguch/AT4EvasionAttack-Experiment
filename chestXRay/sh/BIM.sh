weps1=0.025
python src/BIM.py --AK=white --eps=${weps1}  --output_path=output/BIM/white > output/BIM/white/eps${weps1}.txt

weps2=0.05
python src/BIM.py --AK=white --eps=${weps2}  --output_path=output/BIM/white > output/BIM/white/eps${weps2}.txt

peps1=0.025
python src/BIM.py --AK=proxy --eps=${peps1}  --output_path=output/BIM/proxy > output/BIM/proxy/eps${peps1}.txt

peps2=0.05
python src/BIM.py --AK=proxy --eps=${peps2}  --output_path=output/BIM/proxy > output/BIM/proxy/eps${peps2}.txt
