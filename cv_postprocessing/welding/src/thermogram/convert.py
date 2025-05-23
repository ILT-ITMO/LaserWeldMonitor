import pandas as pd
import matplotlib.pyplot as plt
data = pd.DataFrame()
data_waves = pd.DataFrame()
data_total = pd.DataFrame()
for case in [10,11,12,13,14]:
    file = pd.read_excel(fr'data\weld_project\2. EXPERIMENTS\2.2 RESULTS\2.2.6 SPECTROSCOPY\2023-0101 Spectroscopy\{case}\{case}.xlsx').drop([0,1,2,3,4],axis=0)
    vectors = []
    wave_len = file.iloc[:,0].tolist()
    for i in range(1,len(file.columns)):
        vectors+=file.iloc[:,i].tolist()
    data[f'{case}'] = pd.Series(vectors)
    data_waves[f'{case}'] = pd.Series(wave_len)

    data_total[f'{case}']=pd.Series(vectors)
    data_total[f'{case}_wave']=pd.Series(wave_len*int(len(vectors)/len(wave_len)))

data_total.to_csv(r'D:\Projects\Transformers\svarka\data/total_df.csv',index=False)
data.to_csv(r'D:\Projects\Transformers\svarka\data/vectors_df.csv',index=False)
data_waves.to_csv(r'D:\Projects\Transformers\svarka\data/waves_df.csv',index=False)
# plt.plot(vectors)
# plt.show()

# plt.plot(wave_len,file.iloc[:,1].tolist())
# # plt.plot(wave_len,file.iloc[:,2].tolist())
# # plt.plot(wave_len,file.iloc[:,3].tolist())
# # plt.plot(wave_len,file.iloc[:,4].tolist())
# plt.show()
print()