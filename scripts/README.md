替换系统中/etc/update-motd.d/01-orangepi-startup-text

显示内容如下：
```
  ___                                    ____   _      _     ___   ____              
 / _ \  _ __  __ _  _ __    __ _   ___  |  _ \ (_)    / \   |_ _| |  _ \  _ __  ___  
| | | || '__|/ _` || '_ \  / _` | / _ \ | |_) || |   / _ \   | |  | |_) || '__|/ _ \ 
| |_| || |  | (_| || | | || (_| ||  __/ |  __/ | |  / ___ \  | |  |  __/ | |  | (_) |
 \___/ |_|   \__,_||_| |_| \__, | \___| |_|    |_| /_/   \_\|___| |_|    |_|   \___/ 
                           |___/                                                     
Welcome to Orange Pi Ai Pro
This system is based on Ubuntu 22.04.3 LTS (GNU/Linux 5.10.0+ aarch64)

Welcome to Ubuntu 22.04.3 LTS with Linux 5.10.0+

System load:   17%             Up time:       7 hours, 46 minutes Local users:   3              
Memory usage:  7% of 15Gi      IP:            192.168.9.7    
CPU usage:     4%              Usage of /:    22% of 235G    
```
Orange Pi Ai Pro的主板使用的是华为海思 HiSilicon Ascend 310 或 Ascend 310B AI处理器,
Ubuntu对系统的适配不是太好，获得不了CPU的温度。

