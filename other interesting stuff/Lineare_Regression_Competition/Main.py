# -*- coding: utf-8 -*-
import numpy as np
import os #um den Nutzerpfad rauszufinden über os.getcwd()
import data as dc
import algorithms as al #Algorithmen zur Regressionsberechnung
import imp
import plotter as pt #wird entfernt
import sys #zum vorzeitigen Terminieren des Programs


imp.reload(dc)
imp.reload(al)
imp.reload(pt)#wird entfernt

def Cast2Int(s):
	try:
		int(s)
		return True
	except ValueError:
		return False


class menu_class(object):
    
    #Klassenvariablen
    current_regressiontype=0 #Bestimme Regression
    settings=[[0] for i in range(15)]
    settings[0][0]=3
    settings[10][0]=1 #k fold validation

 
    
#    path= os.getcwd()+"\\"
#    path="C:\Users\Tristan_local\Desktop\\"
    path="C:\\Users\Tristan_local\\Desktop\\Lineare_Regression\\"
    x_name="comp_trainX.dat"
    y_name="comp_trainY.dat"

    dataobject=dc.data_class() 
    
    def run_simulation(self):
        self.read_data()
        tempX=self.dataobject.getXdata()
        tempY=self.dataobject.getYdata()   
 
        regression_function= al.linear_regression_algorithms.regressiontype(self.current_regressiontype, self.dataobject,self.settings)     

        
        #Lösche den folgenden Part außer sys.exit() nachdem Aufgabe abgeschickt wurde:
        newXdata=np.loadtxt(self.path+"comp_testX.dat")
        
        temp=np.empty([len(newXdata)])

        for i in range (len(newXdata)):
            temp[i]=regression_function(newXdata[i])
        np.savetxt(self.path+"comp_testY_PREDICTIONS.dat",temp, delimiter='\n')
        tempobject=dc.data_class()
        tempobject.changeXdata(np.loadtxt(self.path+"comp_testX.dat")) 
        tempobject.changeYdata(np.loadtxt(self.path+"comp_testY.dat"))  
        print("real R^2 on test data:",al.linear_regression_algorithms.r_squared(tempobject, regression_function))
        #sys.exit()
    
    def type_of_regression():
        return        
    
    def change_parameters_menu():
        return
    
    def read_data(self):
        #self.x_data=np.loadtxt(self.path+self.x_name)
        #self.y_data=np.loadtxt(self.path+self.y_name)
        self.dataobject.changeXdata(np.loadtxt(self.path+self.x_name))
        self.dataobject.changeYdata(np.loadtxt(self.path+self.y_name))
    
    def change_path(self):
        schranke=True
        
        while (schranke==True):
            name = input("####\nGeben Sie den vollständigen Pfad als String an (also Anführungszeichen nicht vergessen):")
            if(Cast2Int(name)==True):
                print("Fehler: Es werden nur Strings als akzeptiert.")
                continue
            else:
                self.path=name
                schranke=False
                print("Pfad erfolgreich geändert!")

    
    def change_xname(self):
        schranke=True
        while (schranke==True):
            name=input("####\nGeben Sie den neuen Dateinamen (inklusive .dat) der x-Werte als String an (also Anführungszeichen nicht vergessen):")
            if(Cast2Int(name)==True):
                    print("Fehler: Es werden nur Strings als akzeptiert.")
                    continue
            else:
                schranke=False         
        self.x_name=name
        print("Dateiname erfolgreich geändert!")
    
    def change_yname(self):
        schranke=True
        while (schranke==True):
            name=input("####\nGeben Sie den neuen Dateinamen (inklusive .dat) der y-Werte als String an (also Anführungszeichen nicht vergessen):")
            if(Cast2Int(name)==True):
                    print("Fehler: Es werden nur Strings als akzeptiert.")
                    continue
            else:
                schranke=False         
        self.y_name=name
        print("Dateiname erfolgreich geändert!")
    
    def load_data_menu(self):
        schranke=True
        options={1:self.change_path, 2:self.change_xname,3:self.change_yname,4:self.menu_options}
        while (schranke==True):
            print("####\n\nSuche nach "+self.x_name+" und "+self.y_name+" in "+self.path)
            name=input("1:\tÄndere Pfad\n2:\tÄndere Namen der Datei mit den x-Werten (Realisierungen)\n3:\tÄndere Namen der Datei mit den y-Werten\n4:\treturn")
            if(Cast2Int(name)==False):
                    print("Fehler: Es werden nur Integer als Input akzeptiert.")
                    continue
            else:
                name=int(name)
                schranke=False               
        options[name]()

    def miscellaneous_options(self):
        schranke=True
        if (len(self.dataobject.getXdata())==0):
            print("Suche nach Daten im Verzeichnis "+str(self.path))

            self.read_data()

        options={1:self.run_pearson_product_moment,4:self.nothing}
        while (schranke==True):
            name = input("####\n1:\tPerson Product Moment\n4:\treturn")
            if(Cast2Int(name)==False):
                    print("Fehler: Es werden nur Integer als Input akzeptiert.")
                    continue
                
            else:
                name=int(name)
                schranke=False     
        options[name]()        
    
    def run_pearson_product_moment(self):
        for i in range (len(self.dataobject.getXdata()[0])):
            if(isinstance(self.dataobject.getYdata()[0],(list,np.ndarray))==True):        
                for j in range (len(self.dataobject.getYdata()[0])):
                    moment=al.linear_regression_algorithms.pearson_product_moment(self.dataobject.getXdata()[:,i],self.dataobject.getYdata()[:,j])
                    print("Person Product Moment Y_"+str(j)+" und X_"+str(i)+":"+str(moment))
            elif (isinstance(self.dataobject.getYdata()[0],( int, float ))==True): 
                moment=al.linear_regression_algorithms.pearson_product_moment(self.dataobject.getXdata()[:,i],self.dataobject.getYdata())
                print("Person Product Moment Y_0 und X_"+str(i)+":"+str(moment))                          
        #sys.exit()
        
    def menu_options(self):
        schranke=True

        options={1:self.type_of_regression_menu, 2:self.change_parameters_menu,3:self.load_data_menu,4:self.nothing}
        while (schranke==True):
            name = input("####\n\n1:\tChange Type of regression\n2:\tChange Regression Parameters\n3:\tLoad Data\n4:\treturn")
            if(Cast2Int(name)==False):
                    print("Fehler: Es werden nur Integer als Input akzeptiert.")
                    continue
                
            else:
                name=int(name)
                schranke=False     
        options[name]()
    
    def type_of_regression_menu(self):
        schranke=True
        options={1:self.choose_polynomial_basis_functions}
        while (schranke==True):
            name=input("####\nWähle:\n\n1:Polynomial Basis Functions")
            if(Cast2Int(name)==False):
                print("Fehler: Es werden nur Integer als Input akzeptiert.")
                continue
            else:
                name=int(name)
                schranke=False
        options[name]()  
          
    def choose_polynomial_basis_functions(self):
        current_regressiontype=0
        print("Die Regression findet nun mit Polynomial Basis Functions statt (default).")    

    #Verhindert, dass multiple Objektinstanzen erstellt werden, wenn man den return Befehl ins Hauptmenu, also self.menu() nutzt                            
    def nothing(self):
        print("####\n")
        return              
        
    def change_parameters_menu(self):
        schranke=True
        options={1:self.polynomial_basis_functions_parameters, 2:self.activate_kfold}
        while (schranke==True):
            name=input("####\nÄndere:\n\n1:Polynomial Basis Functions Parameter\n2:(De)activate Crossvalidation")
            if(Cast2Int(name)==False):
                print("Fehler: Es werden nur Integer als Input akzeptiert.")
                continue
            else:
                name=int(name)
                schranke=False
        options[name]()
      
	  
    def activate_kfold(self):
        schranke=True
        print("####\n")
        if  self.settings[10][0]==1:
	         print("Crossvalidation is currently turned on.")
        else:
	         print("Crossvalidation is currently turned off")
        while (schranke==True):
            name=input("0:Turn Crossvalidation off\n1:Turn Crossvalidation on")

            if(Cast2Int(name)==False):
                print("Fehler: Es werden nur Integer als Input akzeptiert.")
                continue
            else:
                name=int(name)
                if (self.settings[10][0]==name):
                    print("\nCrossvalidation settings remain unchanged")
                else:
                    print("\nCrossvalidation successfully been changed!")
				
                self.settings[10][0]=name

                schranke=False
        
        self.nothing()
	
    def polynomial_basis_functions_parameters(self):
        options={1:self.pbf_p1, 2:self.menu_options}
        schranke=True
        while (schranke==True):
            name=input("####\nAktuelle Anzahl der Basisfunktionen:"+str(self.settings[0])+"\n1:Ändere Anzahl der Basisfunktionen\n2\treturn")
            if(Cast2Int(name,)==False):
                print("Fehler: Es werden nur Integer als Input akzeptiert.")
                continue
            else:
                name=int(name)
                schranke=False
        options[name]()
    
    #Ändere Anahl der Basisfunktionen im polynomiellen Basisfunktionsregressionsmodell            
    def pbf_p1(self):
        schranke=True
        while (schranke==True):
            name=input("####\nGeben Sie die neue Anzahl von Basisfunktionen ein: ")
            if(Cast2Int(name)==False):
                print("Fehler: Es werden nur Integer als Input akzeptiert.")
                continue
            elif (int(name)<=0):
                print("Fehler: Die Anzahl der Basisfunktionen muss mindestens 1 betragen")
                continue
            else:
                name=int(name)
                self.settings[0][0]=name
                print("Anzahl der Basisfunktionen erfolgreich geändert!"+str(self.settings[0]))
                schranke=False
        self.menu_options()

 #Das Hauptmenu wird hier definiert       
    def menu(self):
        options= {1:self.run_simulation,3:self.miscellaneous_options, 4:self.menu_options}
        schranke=True
        while (schranke==True):
            name = input("1:\t\033[1mRun Regression\033[0m\n3:\tMiscellaneous\n4:\tOptionen\n0\tExit")
        
            if(Cast2Int(name)==False):
                print("Fehler: Es werden nur Integer als Input akzeptiert.")
                continue
				
            name=int(name)
			
            if (name==0):

                break;
            
            else:
                options[name]()
        
        print("Menu geschlossen!")
        
    #Objekt löschen (TODO)
    def __del__(self):
        del self

    def __init__(self):
          self.menu()

		  
#Rufe Menu auf         
X=menu_class()



