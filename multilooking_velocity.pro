pro velocity_multilooking

  ; first reload the entire scene, extract a test-patch and save this patch to the disk. (later probably done by block processing)
  reload_data = 0
  temp_folder = 'E:\working_dir_IDL\multilook_velocity\'
  out_folder  = 'E:\working_dir_IDL\multilook_velocity\results2\'
  if reload_data then begin
    folder = 'W:\Silvan_Aletsch_coregFeb2017\coreg_2011-2018_bscatter_geo\full_scene_amplitude_rat\'
    files = file_search(folder, '*.rat', count=fcnt)
    for j=0,fcnt-1 do begin
      rrat, files[j], s ; read full scene (I think you could also read in the tiff-files)
      sub = db(abs(s[6500:7500,11000:12000])^2) ; extract patch and save content in dB
      save, sub, filename=temp_folder + file_basename(files[j],'.rat') + '.sav' ; save patch to temporary folder
    end
  end

  ; get all files from the temporary folder which end with *.sav (sav = IDL save file)
  files = file_search(temp_folder+'*.sav', count=fcnt)
  ; restore first file to extract the image dimensions
  restore, files[0] ; -> sub
  sz = size(sub, /dim)

  ; extract date from file name and convert it to julian dates
  times = julday(strmid(file_basename(files),11,2), strmid(file_basename(files),14,2), strmid(file_basename(files),6,4))
  ; select index of first acquisitions (in spring, during snow melt, the glacier surfaces changes a lot. I avoided these acquisitions in April-June)
  jstart = 0;11
  ; set index of last acquisition
  jstop  = 35;24
  ; number of velocities to test
  N = 20
  times = times - times[jstart] ; define first selected scene as zero time reference.

  ; initialize arrays
  comp              = fltarr(sz)
  beststack_avg     = fltarr(sz)
  maxstack_contrast = fltarr(sz)
  est_velocity      = fltarr(sz)

  ; loop to test different velocities
  for k=0,N do begin
    velocity = [-0.25, 0.17]*0.05*k ; here in pixel per day
    print, velocity

    ; load patch from disc
    restore, files[jstart] ; -> creates the variable "sub". The content is in dB
    ; for averaging, the linear backscatter intensity is needed. Therefore convert dB to linear (physically meaningful) units.
    ; initialize the array which contains the average of all images from jstart to jstop
    stack_avg = 10^(sub/10.0)

    ; load all patches
    for j=jstart+1,jstop do begin
      restore, files[j]; -> creates variable "sub"
      print, files[j]
      ; init 2D index array for bilinear interpolation. This allows for sub-pixels shifts
      createmesh, findgen(sz[0]) + velocity[0]*times[j], findgen(sz[1]) + velocity[1]*times[j], xout=IX, yout=IY
      ; shift image j and add it to the averaged stack.
      stack_avg += bilinear(10^(sub/10.0), IX, IY)
    end ; end of image averaging loop.
    ; at the end of each averaging process (for each velocity) show the result of the stack in dB. clip() = autocontrast.
    ;f = fimg(clip(db(stack_avg)))
    ; save image to disk for visual inspection
    ;    f.save, width=sz[0], height=sz[1], out_folder + path_sep() + 'v3=' + strjoin(str(velocity,'%04.2f'),',') + 'px-d.tiff'

    ; calculate intensity of local contrast and compare with best contrast (there might exist a better method?)
    patch_win = 25;
    patch_hp = stack_avg - smooth_gauss(stack_avg, patch_win, /box, /edge_trunc) ; apply high pass to image by subtracting a low pass image
    ; rectify the high-passed image by taking the abs() and blur it a bit.
    weight    = smooth_gauss(abs(patch_hp), patch_win, /box, /edge_trunc)
    ; compare the weight ( = measure for image sharpness) to the currently best value. (pixelwise)
    I = where(weight gt maxstack_contrast) ; I is the index array for which the condition (...) is fulfilled
    ; save the weight ( = meaure for best contrast/image sharpness) for comparison with the other velocity shifts
    maxstack_contrast[I] = weight[I]
    ; add the average image corresponding to the current shift to the final sharpened (temporally multilooked) image product.
    beststack_avg[I] = stack_avg[I]
    ; save current velocity
    est_velocity[I]  = sqrt(velocity[0]^2+velocity[1]^2)

  end; end of testing different velocities

  ; visualize results
  f1 = fimg(clip(db(beststack_avg), th=0.05)) ; create backscatter image of final image.
  f1 = fimg(est_velocity, rgb_table=13, /current, transparency=50, max=0.25) ; add velocity as color overlay to backscatter image
  ; save velocity+backscatter image to disk
  f1.save, width=sz[0], height=sz[1], out_folder + path_sep() + 'backscatter+velocity_j=('+str(jstart,'%i')+'-'+str(jstop)+')_Nvel='+str(N,'%i')+'.tiff'
  f1.close

  ; save backscatter image to disk
  f2 = fimg(clip(db(beststack_avg), th=0.02))
  f2.save, width=sz[0], height=sz[1], out_folder + path_sep() + 'backscatter_j=('+str(jstart,'%i')+'-'+str(jstop)+')_Nvel='+str(N,'%i')+'.tiff'
  f2.close


  stop ; for debugging or interactive modification of the code

end