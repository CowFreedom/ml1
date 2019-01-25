#include <iostream>

extern "C" void newton_banana(double* x){

	int iter=0;
	while(iter<100){
		//double f=(x[0]*x[0])+20*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
		double g1=(2*x[0]-80*(x[1]-x[0]*x[0])*x[0]);
		double g2=x[1]-x[0]*x[0];
		double temp1=40*g1+80*x[0]*g2;
		double temp2=80*x[0]*g1+(-80*x[1]+240*x[0]*x[0]+2)*g2;
		double norm=1/((-80*x[1]+240*x[0]*x[0]+2)*40-6400*x[0]*x[0]);
		//std::cout<<"temp1: "<<40*g1<<"\n";
		x[0]=x[0]-norm*temp1;
		x[1]=x[1]-norm*temp2;
		iter=iter+1;
	}
}