#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<fftw3.h>
#include<omp.h>

#include"nbody.h"

/*------------------------------GLOBAL VARIABLES-------------------------------*/
//-----------------------------------------------------------------------------//
//                              from N-body code 
//-----------------------------------------------------------------------------//

float  vhh, // Hubble parameter in units of 100 km/s/Mpc
  vomegam, // Omega_matter, total matter density (baryons+CDM) parameter
  vomegalam, // Cosmological Constant 
  vomegab, //Omega_baryon
  sigma_8_present, // Last updated value of sigma_8 (Presently PLANCK+WMAP)
  vnn; //Spectral index of primordial Power spectrum

long N1, N2, N3; // box dimension (grid) 
int NF, // Fill every NF grid point 
  Nbin, // Number of bins to calculate final P(k) (output)
  Nmin_count,
  nion_count,
  Nmin; // Minimum halo mass (Mmin in number of DM particle unit) (only for writing outputs)

float LL; // grid spacing in Mpc

long MM; // Number of particles

int zel_flag=1, // memory allocation for zel is 3 times that for nbody
  fourier_flag; //for fourier transfrom
float  DM_m, // Darm matter mass of simulation particle in (10^10 M_sun h^-1) unit 
  norm, // normalize Pk
  pi=M_PI;

io_header header1;

//-----------------------------------------------------------------------------//
//                          from N-body code  done
//-----------------------------------------------------------------------------//



//-----------------------------------------------------------------------------//
//                           needed for N-bdy funcs 
//-----------------------------------------------------------------------------//

float ***ro; // for density
fftwf_plan p_ro; // for FFT
fftwf_plan q_ro; // for FFT

//-----------------------------------------------------------------------------//
//                           needed for N-bdy funcs done  
//-----------------------------------------------------------------------------//

//----------------------arrays for storing ionization data---------------------//

float ***nh, // stores neutral hydrogen on grid points
  ***nhs,    // stores smoothed neutral hydrogen on grid point
  ***ngamma, // stores photon number on grid points
  ***ngammas, // stores smoothed photon number on grid points
  ***nxion;  // stores ionization fractions for different nions on grid points

/*----------------------------GLOBAL VARIABLES DONE----------------------------*/


void main()
{
  long int seed;
  FILE  *inp,*outpp;
  
  int i;
  long ii,jj,iii,jjj,kkk, kk,ll, tmp;
  int sfac;
  
  float vaa;  // final scale factor
  
  //-----------------done variables for non-uniform recombination------------//
  
  double *power_P0, *power_P2, *power_P4, *kmode; // arrays for power spectrum 
  double *no;
  
  float dr,r_min,r_max; // radious for smoothing
  
  char file[200], file1[200], file2[200], num[8], num1[8], num2[8];
  
  float nion,xh1,nion_inp; // to store ionization fraction and neutral hydrogen fraction
  double vion, roion; // to store vol. avg. and mass avg. ionization fraction
  
  int output_flag,in_flag;
  
  long totcluster; // total no. of haloes in halo_catalogue
  float robar,Radii;
  double robarhalo; //no. of dark matter paricle density per (grid)^3
  float vfac; //coefficient for redshift space distortion
  
  float **rra,**vva,**data,**datars, //to store particle positions and velocities
    **halo, **halo_less;
  
  double t,T=omp_get_wtime(); // for timing
  
  int Noutput;
  float *nz, *nions;
  int *Nmins;
  
  
  /*---------------------------------------------------------------------------*/
  /* Read input parameters for the simulation from the file "input.nbody_comp" */
  /*---------------------------------------------------------------------------*/
  
  inp=fopen("input.nbody_comp","r");
  fscanf(inp,"%ld%*d",&tmp);
  fscanf(inp,"%*f%*f%*f%*f");
  fscanf(inp,"%*f%*f");
  fscanf(inp,"%ld%ld%ld%*d%*f",&tmp,&tmp,&tmp);
  fscanf(inp,"%*d%*d");
  fscanf(inp,"%*f%*f");  /* time step, final scale factor*/
  fscanf(inp,"%d%d",&Nmin_count,&nion_count);
  fscanf(inp,"%d",&Noutput);
  
  Nmins=(int*)calloc(Noutput,sizeof(float));
  nions=(float*)calloc(Noutput,sizeof(float));
  nz=(float*)calloc(Noutput,sizeof(float)); // array to store Noutput 
  
  for(i=0;i<Nmin_count;i++)
    fscanf(inp,"%d",&Nmins[i]);
  for(i=0;i<nion_count;i++)
    fscanf(inp,"%f",&nions[i]);
  for(i=0;i<Noutput;i++)
    fscanf(inp,"%f",&nz[i]);
  
  fclose(inp);

  //---------------------------------------------------------------------------//
  //-------------parameters read from input file. Check this ------------------//
  
  sfac=8;
  Nbin=10;
  vion=0.0;
  roion=0.0;

  //---------------------------------------------------------------------------//
  system("mkdir ionz_out");
  /*-----------------------------read nbody output-----------------------------*/

  long newtotcluster=0;

  for(i=0;i<Noutput;i++)
    {
      
      //-------------------------reading the halo catalogue------------------------//
      t=omp_get_wtime();
      //strcpy(file1,"../../../../215mpc_HI_box/run01/recheck/fof/outputs/halo_catalogue_Nmin10_");
      strcpy(file1,"../FoF-Halo-finder-threads/halo_catalogue_");
      sprintf(num1,"%.3f",nz[i]);
      strcat(file1,num1);
          
      read_fof(file1,1,&output_flag,&totcluster,halo,&vaa);
      halo = allocate_float_2d(totcluster,7);
      printf("totcluster=%d\n", totcluster);
      
      read_fof(file1,2,&output_flag,&totcluster,halo,&vaa);
      printf("ok read halo catalogue = %e\n",omp_get_wtime()-t);

      
      // .......................reading sampled N body outputs......................//
      t=omp_get_wtime();
      //strcpy(file,"../../../../215mpc_HI_box/run01/nbody_sampled/sampled.nbody_");
      strcpy(file,"../sampling_general/outputs/sampled.nbody_");
      //strcpy(file,"../calpow/sampled.nbody_");
      sprintf(num,"%.4f",nz[i]);
      strcat(file,num);
    
      read_sampled(file,1,&seed,&output_flag,&in_flag,rra,vva,&vaa); // only read header

      if(i==0)
	{
	  rra = allocate_float_2d(MM,3);
          vva = allocate_float_2d(MM,1);
	  data = allocate_float_2d(MM,5);
	}
      
      read_sampled(file,2,&seed,&output_flag,&in_flag,rra,vva,&vaa); // read data
      
      printf("ok read nbody output = %e\n",omp_get_wtime()-t);
      
      //------------------------------Reading Done---------------------------------//
      
      //-----------------------------Redefine grid---------------------------------//
        
      N1=N1/sfac;  N2=N2/sfac;  N3=N3/sfac;// new grid dimensions 
      LL=LL*sfac; 
      robar=MM*8/(1.*N1*N2*N3); // mean number density (grid)^{-3}
      vfac=1./(Hf(vaa)*vaa*vaa); // for redshift space distortion
      
      //---------------------------------------------------------------------------//
     
      for(ii=0;ii<MM;ii++)
	{
	  data[ii][0] = rra[ii][0]/(1.*sfac);
	  data[ii][1] = rra[ii][1]/(1.*sfac);
	  data[ii][2] = rra[ii][2]/(1.*sfac);
	  
	  data[ii][3] = (rra[ii][2] + vfac*vva[ii][0])/(1.*sfac); // redshift space distortion applied
	  data[ii][3] += N3*1.;
	  data[ii][3] = data[ii][3]-1.0*N3*(int)(floor(data[ii][3])/(1.*N3));
	  
	  data[ii][4] = 8.;  // same mass for all particles
	}
      printf("LL=%f\n",LL);
      /*----------------------------------------------------------------*/    
      
      
      //............From here new changes for EoR parameter multiout...............//
    
    
      if(i==0)
	{
	  Setting_Up_Memory_For_ionz();
          /*---------allocate memory for power spectrum and k modes--------*/
          kmode=calloc((size_t)Nbin,sizeof(double));
          power_P0=calloc((size_t)Nbin,sizeof(double));
          power_P2=calloc((size_t)Nbin,sizeof(double));
          power_P4=calloc((size_t)Nbin,sizeof(double)); 
          no=calloc((size_t)Nbin,sizeof(double));
	}
   
      MM=header1.npart[1];
      printf("MM=%d\n, N1=%d\n", MM, N1);
      cic_vmass(nh, data, 0, 1, 2, 4);
  
      printf("ok cic_vmass of dark matter= %e\n",omp_get_wtime()-t);

      for(iii=0;iii<Nmin_count;iii++)   //Nmin parameter loop start
        {
          Nmin=Nmins[iii];
          printf("Nmin=%d\n",Nmin);

          newtotcluster=0;
          for(ii=0;ii<totcluster;ii++)
            {
              if(halo[ii][0] >= Nmin)
                {
                  newtotcluster++;
                }
            }
          printf("newtotcluster=%ld\n",newtotcluster);
          halo_less = allocate_float_2d(newtotcluster,4);
     
          for(ii=0;ii<totcluster;ii++)
	    { 
              if(halo[ii][0] >= Nmin)   // Nmin is defied by the reionization model parameters
                {
	          halo_less[ii][1] = halo[ii][1]/(1.*sfac);
                  halo_less[ii][2] = halo[ii][2]/(1.*sfac);
                  halo_less[ii][3] = halo[ii][3]/(1.*sfac);	    
	          halo_less[ii][0] = halo[ii][0];
                }
	    }

  
          t=omp_get_wtime();
      
          /* calculating the halo mass density at each grid point */

          MM=newtotcluster;
          cic_vmass(ngamma, halo_less, 1, 2, 3, 0); 
 

          for(jjj=0;jjj<1;++jjj)    //nion loop start
            {  
              nion=nions[jjj];
              printf("nion=%f\n",nion);

      
          //---------------------subgrid re-ionization----------------------//
      
          t=omp_get_wtime();
      
          for(ii=0;ii<N1;ii++)
      	    for(jj=0;jj<N2;jj++)
	      for(kk=0;kk<N3;kk++)
	        {
	          if(nh[ii][jj][kk]>nion*ngamma[ii][jj][kk]) // checking ionization condition
		    {
		      nxion[ii][jj][kk]=nion*ngamma[ii][jj][kk]/nh[ii][jj][kk];
		    }
	      
	          else
		    {
		      nxion[ii][jj][kk]=1.;
		    }
	        }    

      
          //-------------calculating avg. ionization fraction---------------//
      
          /*----------------------------------------------------------------*/
          /*----------------------------------------------------------------*/
          //calculating max and min radius for smoothing in units of grid size
      
          r_min=1.;
          r_max=20.0/LL; // Mpc/LL in grid unit    //R_mfp, change this to create different models

          //r_max=pow((1.*N1*N2*N3),(1./3.))/2.;
      
          /*----------------------------------------------------------------*/
          /*                        smoothing                               */
          /*----------------------------------------------------------------*/
      
          t=omp_get_wtime();
      
          Radii=r_min;
      
          while(Radii < r_max)
	    {
	      for(ii=0;ii<N1;ii++)
	        for(jj=0;jj<N2;jj++)
	          for(kk=0;kk<N3;kk++)
		    {
		      nhs[ii][jj][kk]=nh[ii][jj][kk];
		      ngammas[ii][jj][kk]=ngamma[ii][jj][kk];
		    }
	  //printf("starting smoothing for radius of size %e\n",Radii);
	  
	      smooth(nhs,Radii);
	  
	      smooth(ngammas,Radii);
	  
	  
	      for(ii=0;ii<N1;ii++)
	        for(jj=0;jj<N2;jj++)
	          for(kk=0;kk<N3;kk++)
		    {
		      if(nhs[ii][jj][kk]<=nion*ngammas[ii][jj][kk])  // checking ionization condition
		        nxion[ii][jj][kk]=1.;
		    }
	  
	      dr=(Radii*0.1) < 2.0 ? (Radii*0.1) : 2.0; //increment of the smoothing radius
	      Radii += dr;
	    }
      
          printf("ok smoothing = %e\n",omp_get_wtime()-t);
       
      
          t=omp_get_wtime();
      
          //---------------calculating avg. neutral fraction-------------*/

          vion =0.0;
          roion=0.0;
      
          /*--------------------writing the results----------------------*/
      
          strcpy(file2,"ionz_out/HImap_Nion");
          sprintf(num2,"%.2f",nion);
          strcat(file2,num2);
          strcat(file2, "_Mmin");
          sprintf(num2,"%d",Nmin);
          strcat(file2,num2);
          strcat(file2,"_z");
          sprintf(num2,"%.4f",nz[i]);
          strcat(file2,num2);
          outpp=fopen(file2,"w");
      
          fwrite(&N1,sizeof(int),1,outpp);
          fwrite(&N2,sizeof(int),1,outpp);
          fwrite(&N3,sizeof(int),1,outpp);

          for(ii=0;ii<N1;ii++)
	    for(jj=0;jj<N2;jj++)
	      for(kk=0;kk<N3;kk++)
	        {
  	          xh1=(1.-nxion[ii][jj][kk]);
  	          xh1=(xh1 >0.0)? xh1: 0.0;
	      
  	          nxion[ii][jj][kk]=xh1; // store x_HI instead of x_ion
  	          nhs[ii][jj][kk]=xh1*nh[ii][jj][kk]; // ro_HI on grid
  	          vion+=(double)xh1;
 	          roion+=(double)nhs[ii][jj][kk];
	           
	          fwrite(&nhs[ii][jj][kk],sizeof(float),1,outpp);
	        }
      
          fclose(outpp);
      
          /*----------------------------------------------------------------*/
      
          roion/=(1.*N1*N2*N3); // mean HI density
      
          /*----------------------------------------------------------------*/
      
          calpow_mom(nhs,Nbin,power_P0,kmode,power_P2,power_P4,no); // calculates moments of redshift space power spectrum
      
          strcpy(file2,"ionz_out/pk.ionz_Nion");
          sprintf(num2,"%.2f",nion);
          strcat(file2,num2);
          strcat(file2, "_Mmin");
          sprintf(num2,"%d",Nmin);
          strcat(file2,num2);
          strcat(file2,"_xHI");
          sprintf(num2,"%.4f",roion/robar);
          strcat(file2,num2);
          strcat(file2,"_z");
          sprintf(num2,"%.4f",nz[i]);
          strcat(file2,num2);
      
          outpp=fopen(file2,"w");
      
          for(ii=0;ii<Nbin;++ii)
	    {
	      power_P0[ii]/=(roion*roion);
	      power_P2[ii]/=(roion*roion);
	      power_P4[ii]/=(roion*roion);
	  
	      fprintf(outpp,"%e %e %e %e %ld\n",kmode[ii],power_P0[ii],power_P2[ii],power_P4[ii],(long)no[ii]);
	    }
      
          fclose(outpp);
      
          /*----------------------------------------------------------------*/
      
          vion/=(1.*N1*N2*N3); // volume avg xHI
          roion/=robar; // divide by H density to get mass avg. xHI
      
          printf("vol. avg. x_HI=%e, mass avg. x_HI=%e\n",vion,roion);
      
      
          /*----------------------------------------------------------------*/
          /*            Do the same for redshift space                      */
          /*----------------------------------------------------------------*/
          
	  MM=header1.npart[1];
          //printf("MM=%d\n", MM);
          for(kkk=0;kkk<MM;kkk++)
            data[kkk][4]=1.;
          density_2_mass(nxion, data, 0, 1, 2, 4);   // get particles HI masses from HI density 
          //printf("MM=%d\n", MM);
          cic_vmass(nhs, data, 0, 1, 3, 4);  // convert particles HI masses to  HI density
      
          printf("ok cic_vmass redshift space = %e\n",omp_get_wtime()-t);
          //------------calculating avg. ionization fraction----------------//
      
          roion=0.0;
      
          strcpy(file2,"ionz_out/HImaprs_Nion");
          sprintf(num2,"%.2f",nion);
          strcat(file2,num2);
          strcat(file2, "_Mmin");
          sprintf(num2,"%d",Nmin);
          strcat(file2,num2);
          strcat(file2,"_z");
          sprintf(num2,"%.4f",nz[i]);
          strcat(file2,num2);
          outpp=fopen(file2,"w");
     
          fwrite(&N1,sizeof(int),1,outpp);
          fwrite(&N2,sizeof(int),1,outpp);
          fwrite(&N3,sizeof(int),1,outpp);

          for(ii=0;ii<N1;ii++)
	    for(jj=0;jj<N2;jj++)
	      for(kk=0;kk<N3;kk++)
	        { nhs[ii][jj][kk]=nhs[ii][jj][kk]*8.;    //for sampling
	          roion+=(double)nhs[ii][jj][kk];
	          fwrite(&nhs[ii][jj][kk],sizeof(float),1,outpp);
	        }
          fclose(outpp);
          roion/=(1.*N1*N2*N3); // mean HI density
      
          /*----------------------------------------------------------------*/
      
          calpow_mom(nhs,Nbin,power_P0,kmode,power_P2,power_P4,no); // calculates moments of redshift space power spectrum
      
          strcpy(file2,"ionz_out/pk.ionzs_Nion");
          sprintf(num2,"%.2f",nion);
          strcat(file2,num2);
          strcat(file2, "_Mmin");
          sprintf(num2,"%d",Nmin);
          strcat(file2,num2);
          strcat(file2,"_xHI");
          sprintf(num2,"%.4f",roion/robar);
          strcat(file2,num2);
          strcat(file2,"_z");
          sprintf(num2,"%.4f",nz[i]);
          strcat(file2,num2);
      
          outpp=fopen(file2,"w");
      
          for(ii=0;ii<Nbin;++ii)
	    {
	      power_P0[ii]/=(roion*roion);
	      power_P2[ii]/=(roion*roion);
	      power_P4[ii]/=(roion*roion);
	  
	      fprintf(outpp,"%e %e %e %e %ld\n",kmode[ii],power_P0[ii],power_P2[ii],power_P4[ii],(long)no[ii]);
	    }
          fclose(outpp);
      
          /*----------------------------------------------------------------*/
      
          roion/=robar; // divide by H density to get mass avg. xHI
          printf("mass avg. x_HI=%e\n",roion);

      
      
           }//end of nion loop
          free(halo_less);
        }//end of Nmin loop     

      printf("ok time taken= %e\n\n",omp_get_wtime()-t);
      
      free(halo);
    }// end of redshift (nz) loop

  free(rra);
  free(vva);
  free(data);
  free(ro);
  free(nh);
  free(nhs);
  free(ngamma);
  free(ngammas);
  free(nxion);


    
  printf("done. Total time taken = %dhr %dmin %dsec\n",(int)((omp_get_wtime()-T)/3600), (int)((omp_get_wtime()-T)/60)%60, (int)(omp_get_wtime()-T)%60);
}



