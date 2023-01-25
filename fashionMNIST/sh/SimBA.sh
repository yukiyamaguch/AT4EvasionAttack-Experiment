query1=100
query2=300
query3=500

# adv_train, proxy
python src/SimBA.py --AK=robust_white --query=${query1}  --output_path=output/SimBA/robust_white > output/SimBA/robust_white/query${query1}.txt
python src/SimBA.py --AK=robust_white --query=${query2}  --output_path=output/SimBA/robust_white > output/SimBA/robust_white/query${query2}.txt
python src/SimBA.py --AK=robust_white --query=${query3}  --output_path=output/SimBA/robust_white > output/SimBA/robust_white/query${query3}.txt
