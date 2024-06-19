#import os
#import sys
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd

plt.rcParams['text.usetex'] = True

#import matplotlib.colorbar as colorbar

##########################################################################
z=8   #redshift, put the value here
Nmin=np.array([10]) #, 15])   #, 20, 25])
nion=np.array([23.21]) #, 26.54   #, 29.88, 33.21])
space=['map','maprs']
for M in space:
 for Mmin in Nmin:
  for Nion in nion:

    

    #Mmin=20  #10, 15, 20, 25
    #Nion=26.54 # 23.21, 26.54, 29.88, 33.21


    filename = "ionz_out/HI{}_Nion{:.2f}_Mmin{}_z8.0".format(M,Nion,Mmin)
    #filename = "ionz_out/HImaprs_Nion{:.2f}_Mmin{}_z8.0".format(Nion,Mmin)



    f = open(filename)
    temp_mesh = np.fromfile(f,count=3,dtype='int32')
    mesh_x, mesh_y, mesh_z = temp_mesh
    datatype = dtype=(np.float32)

    HI_maplc = np.fromfile(f, dtype=datatype,count=mesh_x*mesh_y*mesh_z)
    f.close()

    print(mesh_x, mesh_y, mesh_z)

    avg_xh1 = np.mean(HI_maplc)
    print("Average x_HI =", avg_xh1/64)
    avg_xh1 = np.sum(HI_maplc)/(mesh_x*mesh_y*mesh_z*64)
    print("Average x_HI =", avg_xh1)

    var_xh1 = np.var(HI_maplc)
    print(var_xh1)

    HI_maplc = HI_maplc.reshape((mesh_x, mesh_y, mesh_z), order='C')

    HI_maplc = HI_maplc*22.*np.sqrt((1.+z)/7.)/64. # Important! For coeval cube, off for lightcone 



##########################################################################

    v_min = np.amin(HI_maplc)
    print(v_min)
    vmin = 0.0

    v_max = np.amax(HI_maplc)
    print(v_max)
    v_max = 75

    fig = pl.figure(1.0, (5.,5.))

##########################################################################

    grid = ImageGrid(fig, 111, # similar to subplot(111)
                nrows_ncols = (1,1), # creates mxn grid of axes
                axes_pad=0.119, # pad between axes in inch.
                label_mode = "L",
                share_all = False,
                cbar_location = "right",
                cbar_mode="single",
                cbar_pad = 0.035,
                cbar_size = "2.0%",
                )


    im = grid[0].imshow(HI_maplc[:,21,:],extent=[0,215.04,0,215.04],vmin=v_min,vmax=v_max) #,cmap='afmhot')


    grid[0].imshow(HI_maplc[:,21,:],extent=[0,215.04,0,215.04],vmin=v_min,vmax=v_max) #,cmap='afmhot')
    grid[0].set_ylabel("Mpc",fontname="Times New Roman",fontsize="20")
    grid[0].set_xlabel("Mpc",fontname="Times New Roman",fontsize="20")

    props = dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='k')
    #plt.text(0.98, 0.98, r"$z={:.4f}$".format(z), ha='right', verticalalignment='top', bbox=props, fontsize=12,       transform=plt.gca().transAxes)

    plt.text(0.98, 0.98, r"$ {:.4f}$".format(avg_xh1), ha='right', verticalalignment='top', bbox=props, fontsize=28, transform=plt.gca().transAxes)

    #plt.text(0.98, 0.98, r"$z=9.3$", ha='right', verticalalignment='top', bbox=props, fontsize=12, transform=plt.gca().transAxes)

    #grid[0].set_xlabel("$z$",fontname="Times New Roman",fontsize="14.5")
    #grid[0].set_ylabel("Grids",fontname="Times New Roman",fontsize="14.5")
    #grid[0].set_xlabel("Grids",fontname="Times New Roman",fontsize="14.5")
    grid[0].grid(True, color='white', linestyle='--', linewidth=0.25)


    #fig.canvas.draw()

    #labels = [item.get_text() for item in grid[0].get_xticklabels()]
    #labels[0] = '7.20'
    #labels[1] = '7.35'
    #labels[2] = '7.66'
    #labels[3] = '8.00'
    #labels[4] = '8.34'
    #labels[5] = '8.69'
    #grid[0].set_xticklabels(labels)

##########################################################################

    plt.colorbar(im, cax=grid.cbar_axes[0])

    #grid[0].cax.colorbar(im)
    cax = grid.cbar_axes[0]
    axis = cax.axis[cax.orientation]
    axis.label.set_text("$T_{b}\,\,($mK$)$")

##########################################################################

    plt.savefig("maps_plots/HI{}_Nion{:.2f}_Mmin{}_z8".format(M,Nion,Mmin), format='png', dpi=300, transparent=False, bbox_inches='tight')
#plt.savefig("maps_plots/HI_maprs_Nion26.54_{:.3f}.png".format(z), format='png', dpi=300, transparent=False, bbox_inches='tight')


#pl.savefig('lcmaprs_7.9722_test.png',bbox_inches='tight')
#pl.savefig('maps_7.9847.png',bbox_inches='tight')

#plt.savefig("143Mpc_maps_plots/lcmaprs_test_9.3.png", format='png', dpi=300, transparent=False, bbox_inches='tight')

