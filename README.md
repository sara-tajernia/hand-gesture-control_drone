# هدايت پهپاد با علائم دست مبتني بر بينايي ماشين

هدف اصلی پیاده‌سازی این پروژه توسعه سیستمی است که بتواند با دقت بالا و زمان کوتاه توسط دوربین علامت دست کاربر را تشخیص داده و پهپاد را با کمک دستور داده شده کنترل کند. 

## ویدیو پیاده‌سازی شده پروژه بر روی پهپاد dji tello


<div align="center">
  <video src="https://github.com/sara-tajernia/hand-gesture-control_drone/assets/61985583/8b8a892d-248e-404c-ae68-584b611fcbbe" width="400" />
</div>

## معرفی
این پروژه بر دو بخش اصلی تکیه دارد - پهپاد DJI Tello و تشخیص علامت دست
پهپاد dji tello یک بستر عالی برای هر نوع آزمایش برنامه نویسی است. این پهپاد دارای API پایتون است که توانایی کنترل کامل پهپاد را دارد.

موارد مورد نیاز برای اجرای پروژه:

<p align="center">
 <img width="40%" alt="flowchart" src="https://github.com/sara-tajernia/hand-gesture-control_drone/assets/61985583/c349e013-3049-4832-a11b-3c9681464b4a">
</p>


## پیاده‌سازی

### نصب package های مورد نیاز
برای اجرای این برنامه باسد از پایتون 3.7 به بالا استفاده کرد.
List of packages
```sh
ConfigArgParse == 1.2.3
djitellopy == 1.5
numpy == 1.19.3
opencv_python == 4.5.1.48
tensorflow == 2.4.1
mediapipe == 0.8.2
```

Install
```sh
pip3 install -r requirements.txt
```

### روشن کردن پهپاد و وصل شدن به آن از طریق وای فای

<p align="left">
<img width="30%" alt="110932822-a7b30f00-8334-11eb-9759-864c3dce652d-2" src="https://github.com/sara-tajernia/hand-gesture-control_drone/assets/61985583/7efe5c6b-6518-40ff-9b1f-2a810b7774c7">
</p>


## روند اجرای پروژه 
<p align="center">
 <img width="60%" alt="flowchart" src="https://github.com/sara-tajernia/hand-gesture-control_drone/assets/61985583/86605a9c-a743-4c73-8a77-0e7569c4ff2a">
</p>

### مدل تشخیص کف دست



### مدل نشخیص نقاط عطف دست
<p align="center">
<img width="1073" alt="hand-landmarks" src="https://github.com/sara-tajernia/hand-gesture-control_drone/assets/61985583/1709cbbc-6a18-435d-952f-bfdebea73b3b">
</p>

### تشخیص علامت دست 

<p align="center">
<img width="1073" alt="hand-landmarks" src="https://github.com/sara-tajernia/hand-gesture-control_drone/assets/61985583/5189b6ff-8463-4368-b92e-1444243bd466">
</p>

### قابلیت‌های برنامه
### توانایی اضافه کردن داده جدید به مجموعه داده با تنها یک دکمه برای اضافه کردن علامت دست جدید
### قابلیت انتخاب کنترل پهپاد با دست راست پا چپ

## نویسنده
سارا تاجرنیا (https://github.com/sara-tajernia)

## استاد راهنما
دکتر مهدی جوانمردی
