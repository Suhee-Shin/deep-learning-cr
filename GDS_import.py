import os
import numpy as np
import matplotlib.pyplot as plt
import lumapi
import time

# 경로 설정
fdtd_file = r"C:/Users/Suhee/Desktop/gc optimize/gc outcoupler_v2.fsp"
gds_folder = r"C:/Users/Suhee/Desktop/gc optimize/gs_algorithm2/output_structures/GDS"
gds_prefix = "S"
num_files = 50001  # 마지막 숫자 +1

# FDTD 실행
fdtd = lumapi.FDTD()
results_dB = []
    
for idx in range(num_files):
    gds_name = f"{gds_prefix}{idx:05d}.gds"
    gds_path = os.path.join(gds_folder, gds_name)
    cellname = f"{gds_prefix}{idx:05d}"
    layer = "1:0"
    wg_mat = "Si (Silicon) - Palik"
    z_height = 220e-9
    
    if not os.path.exists(gds_path):
        print(f"파일 없음: {gds_path}, 건너뜀")
        continue
    
    print(f"[{idx}] GDS 불러오는 중: {gds_name}")
    
    try:
        fdtd.load(fdtd_file)
        time.sleep(0.5)  # 로딩 안정화 대기
        fdtd.eval('switchtolayout;')

        fdtd.eval(f'''
    gdsimport("{gds_path}", "{cellname}", "{layer}", "{wg_mat}", 0, {z_height});
    #set("name","GDS");
    ''')
    
        ## GDS 구조물 설정
        fdtd.select("GDS_LAYER_0:0")
        fdtd.set("x", 14e-6)
        fdtd.set("y", 8.4e-6)
        fdtd.set("z", 0.11e-6)
        

    except Exception as e:
        print(f"[{idx}] 파일 삽입 실패: {e}")
        results_dB.append(np.nan)
        continue
    try:
        fdtd.run()
        
        ## 결과 추출
        result = fdtd.getresult("mode", "expansion for output_T")
        
        if result and "T_backward" in result:
            T = np.real(np.array(result["T_backward"]).squeeze())   # shape = (100, 100)
            #value = 10 * np.log10(np.abs(T))
            value = (np.abs(T))

            # numpy 파일로 저장
            save_path = os.path.join(r"C:/Users/Suhee/Desktop/gc optimize/gs_algorithm2/output_structures/results_v3", f"result_{idx+90000:05d}.npy")
            np.save(save_path, value)

            print(f"[{idx}] 완료: 행렬 {value.shape} 저장됨 -> {save_path}")

        else:
            print(f"[{idx}] 결과 없음")
       
        fdtd.eval("select('GDS_LAYER_0:0');")
        fdtd.eval('switchtolayout;')
        fdtd.eval('delete;')
        fdtd.eval('save;')

        


    except Exception as e:
        print(f"[{idx}] 시뮬레이션 실패: {e}")
        results_dB.append(np.nan)
        continue
        
print("모든 시뮬레이션 완료")