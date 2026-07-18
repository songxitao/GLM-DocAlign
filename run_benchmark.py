import sys
import os
import time
import subprocess
from pathlib import Path

# PDF 路径硬编码或从参数读取
DEFAULT_PDF = r"E:\desktop\code\New folder\44231566_SONG XITAO（2）.pdf"
PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF

def run_cmd(cmd, cwd=None):
    # 实时输出子进程 stdout，防止大日志被阻塞憋死
    process = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="ignore")
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    return process.poll()

def main():
    pdf = Path(PDF_PATH)
    if not pdf.exists():
        print(f"[FAIL] 找不到测试 PDF 文件: {pdf}")
        print("请检查路径是否正确，或者在命令行参数中传入正确的路径，例如:")
        print('python run_benchmark.py "E:\\path\\to\\your.pdf"')
        sys.exit(1)
        
    python_exe = sys.executable
    print("=" * 60)
    print("        GLM-OCR 性能对照测试 (Benchmark)")
    print("=" * 60)
    print(f"测试文件: {pdf}")
    print(f"当前 Python: {python_exe}")
    print("-" * 60)
    
    # 步骤一：切回老版本
    print("\n[1/4] 正在物理切回老串行版本 (git checkout -f HEAD~1)...")
    rc = subprocess.run("git checkout -f HEAD~1", shell=True, capture_output=True, text=True)
    if rc.returncode != 0:
        print(f"[FAIL] Git 切换老版本失败: {rc.stderr}")
        print("请确保在 GLM-OCR git 仓库主目录下运行此脚本。")
        sys.exit(1)
        
    # 步骤二：计时运行老版本
    print("\n[2/4] 正在运行老串行版本管线，开始计时...")
    t1_start = time.time()
    cmd_run = f'"{python_exe}" run_pipeline.py "{pdf}" --force'
    run_cmd(cmd_run)
    t1_end = time.time()
    t1_duration = t1_end - t1_start
    print(f"[INFO] 老版本运行完毕，总耗时: {t1_duration:.2f} 秒。")
    
    # 步骤三：切回新版本
    print("\n[3/4] 正在物理切回新异步流水线版本 (git checkout -f main)...")
    rc = subprocess.run("git checkout -f main", shell=True, capture_output=True, text=True)
    if rc.returncode != 0:
        print(f"[FAIL] Git 切回主分支失败: {rc.stderr}")
        sys.exit(1)
        
    # 步骤四：计时运行新版本
    print("\n[4/4] 正在运行新异步流式管线，开始计时...")
    t2_start = time.time()
    run_cmd(cmd_run)
    t2_end = time.time()
    t2_duration = t2_end - t2_start
    print(f"[INFO] 新异步版本运行完毕，总耗时: {t2_duration:.2f} 秒。")
    
    # 步骤五：打印评估报告
    print("\n" + "=" * 60)
    print("                性能评估对照报告")
    print("=" * 60)
    print(f"老版本耗时: {t1_duration:.2f} 秒")
    print(f"新版本耗时: {t2_duration:.2f} 秒")
    
    diff = t1_duration - t2_duration
    if diff > 0:
        percent = (diff / t1_duration) * 100
        speedup = t1_duration / t2_duration
        print(f"[OK] 提速效果: 新版本比老版本快了 {diff:.2f} 秒！")
        print(f"[INFO] 吞吐效率提升率: {percent:.2f}% (提速比约为 {speedup:.2f}x)")
    else:
        print("[WARN] 未检测到加速，可能是由于单页小文档网络延迟起主导作用，或者大模型队列排队堵塞。")
    print("=" * 60)

if __name__ == "__main__":
    main()
