# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:15:57 2020

@author: Shubham
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
import scadnano as sc


#To-do
#0. DONE - Fix staples being too short - merge short ones with adjacent staples? - Check if this is only on the perimeter?
#1. Optimize rasterization of face (max fraction of area covered among all the rotations may be?)
#2. Scaffold routing through all faces - polyhedron net algorithm? Hamiltonian path of the dual
# polyhedron of our input shape (in other words the face graph) is the answer but that's not always findable 
#- settle with brute force backtracking algo?

#print('h='+str(helix)+' o='+str(offset))

def PolyArea(x,y): 
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))



scaflen = 8064 #scaffold length to be used
block = 14 #controls effective "pixel" width, height is fixed at 2 helices for now

r = np.array([1,50,60,60])
c = np.array([8, 50,24,1])
rp = np.append(r,r[0])
cp = np.append(c,c[0])

mag = np.sqrt((scaflen/(2.0*block))/(1.05*PolyArea(r,c))) #adjusted so scaffold length in origami is shorter than scaflen
# mag = 0.3 #just a magnification factor

print(mag)      
print((mag**2)*PolyArea(r,c))

rr, cc = polygon((mag*c), (mag*r))
rmax = np.max(rr)+2
cmax = np.max(cc)+2
maxhelix = 2*rmax-1
img = np.zeros((rmax, cmax), dtype=int)
img[rr, cc] = 1
img = img[::-1,:] #flip to match plot
# img[[10,11,12]][:,[21,22]] = [0]


#deal with 4-disconnected cells
#Cannot deal with completely disconnected cells

for index, value in np.ndenumerate(img):
    x,y = index
    if x == len(img)-1 or y==len(img[0])-1:
        continue
    if img[x,y] and \
       img[x+1,y+1] and \
       not img[x+1,y] and \
       not img[x,y+1] :
         img[x+1,y] = 1
         img[x,y+1] = 1

    elif not img[x,y] and \
       not img[x+1,y+1] and \
       img[x+1,y] and \
       img[x,y+1] :
         img[index] = 1
         img[x+1,y+1] = 1
         

#To check actual scaffold length
count = 0
for x in np.nditer(img):
    if x == 1:
        count+=1
print(count)
print(count*2*block)

#Plot shape of design
fig = plt.figure()
plt.plot(rp,cp)
# plt.axis('square')
plt.axis('equal')
plt.style.use('dark_background')
plt.show()




def create_design():
    design = precursor_scaffolds()
    add_scaffold_nicks(design)
    add_scaffold_crossovers(design)
    design.strands[0].set_scaffold()
    precursor_staples(design)
    add_staple_nicks(design)
    return design

def domain_end(design: sc.Design, helix, offset): #True if it is a domain end or if nothing exists
    if len(design.domains_at(helix, offset))==2:
        if design.domains_at(helix, offset)[0].start == offset or \
           design.domains_at(helix, offset)[0].end == offset or \
           design.domains_at(helix, offset)[1].start == offset or \
           design.domains_at(helix, offset)[1].end == offset :
               return True
        else :
             return False
    elif len(design.domains_at(helix, offset))==1:
        if design.domains_at(helix, offset)[0].start == offset or \
           design.domains_at(helix, offset)[0].end == offset : 
               return True
        else:
            return False
    else:
        return True #since we don't want crossovers where nothing exists

def precursor_scaffolds() -> sc.Design:
    
    #2D design has to be topologically a circle
    
    helices = [sc.Helix(max_offset=int(block*cmax)) for _ in range(maxhelix)]
    scaffolds = []
    for helix in range(len(img)):
        if (1 in img[helix]) :
            for dom in range(len(img[helix])):
                if (img[helix][dom]==1) and (img[helix][dom-1]==0) :
                    start = block*dom
                if (img[helix][dom]==0) and (img[helix][dom-1]==1) :
                    end = block*dom
        
                    scaffolds.append(sc.Strand([sc.Domain(helix=2*helix,
                        forward=((2*helix) % 2 == 0), start=start, end=end)]))
                    scaffolds.append(sc.Strand([sc.Domain(helix=(2*helix)+1,
                        forward=((2*helix+1) % 2 == 0), start=start, end=end)]))    
    return sc.Design(helices=helices, strands=scaffolds, grid=sc.square)

def precursor_staples(design: sc.Design):
    
    #2D design has to be topologically a circle
    
    # helices = [sc.Helix(max_offset=int(block*cmax)) for _ in range(maxhelix)]
    staples = []
    for helix in range(len(img)):
        if (1 in img[helix]) :
            for dom in range(len(img[helix])):
                if (img[helix][dom]==1) and (img[helix][dom-1]==0) :
                    start = block*dom
                if (img[helix][dom]==0) and (img[helix][dom-1]==1) :
                    end = block*dom

                    staples.append(sc.Strand([sc.Domain(helix=2*helix,
                        forward=((2*helix) % 2 == 1), start=start, end=end)]))
                    staples.append(sc.Strand([sc.Domain(helix=(2*helix)+1,
                        forward=((2*helix+1) % 2 == 1), start=start, end=end)]))
    for staple in staples:
            design.add_strand(staple)
        
def add_staple_nicks(design: sc.Design):
    
    crossovers = []
    
    for helix in range(3, maxhelix, 2): 
        if not design.strands_starting_on_helix(helix):
            pass
        else :
            hel1 = helix #first starting helix
            break
    
    for helix in range(hel1, maxhelix, 2):
        
        scafdom1 = design.strands_starting_on_helix(helix)
        scafdom2 = design.strands_starting_on_helix(helix+1)
        if not (scafdom1 and scafdom2): #check for empty helix at the end of the design
            break
  
    for helix in range(maxhelix):
        if helix%2 == 0 :
            for offset in range(block*cmax):
                if (offset%32==0) and (design.domains_at(helix, offset)):
                    if domain_end(design, helix, offset) : #no nick if domain end
                        continue
                    #Prevent very short staples
                    if (design.domains_at(helix, offset-16)) and \
                       (design.domains_at(helix, offset+16)):
                        design.add_nick(helix=helix, offset=offset, forward=helix % 2 == 1)
                    if 0<=(offset-8)<=block*cmax and \
                       (design.domains_at(helix, offset-16)) and \
                       (design.domains_at(helix, offset)) and \
                       (design.domains_at(helix+1, offset-8)) and \
                       (design.domains_at(helix+1, offset-9)) and \
                       not domain_end(design, helix, offset-8):              
                        crossovers.append(sc.Crossover(helix=helix, helix2=helix + 1, \
                                                   offset=offset-8, forward=helix % 2 == 1))

        else :
            for offset in range(block*cmax):
                if ((offset+16)%32==0) and (design.domains_at(helix, offset)):
                    if domain_end(design, helix, offset) :
                        continue
                    #Prevent very short staples
                    if (design.domains_at(helix, offset-16)) and \
                       (design.domains_at(helix, offset+16)):
                        design.add_nick(helix=helix, offset=offset, forward=helix % 2 == 1)
                    if 0<=(offset-8)<=block*cmax and \
                       (design.domains_at(helix, offset-16)) and \
                       (design.domains_at(helix, offset)) and \
                       (design.domains_at(helix+1, offset-8)) and \
                       (design.domains_at(helix+1, offset-9)) and \
                       not domain_end(design, helix, offset-8):
                           
                        crossovers.append(sc.Crossover(helix=helix, helix2=helix + 1, \
                                                   offset=offset-8, forward=helix % 2 == 1))

    design.add_crossovers(crossovers)

def add_scaffold_nicks(design: sc.Design):
    crossovers = []

    for helix in range(3, maxhelix, 2): 
        if not design.strands_starting_on_helix(helix):
            pass
        else :
            hel1 = helix #first starting helix
            break
    
            
    for helix in range(hel1, maxhelix, 2):
        
        scafdom1 = design.strands_starting_on_helix(helix)
        scafdom2 = design.strands_starting_on_helix(helix+1)
        if not (scafdom1 and scafdom2): #check for empty helix at the end of the design
            break
        
        imghelix = int((helix-1)/2)
        if (1 in img[imghelix]) and (1 in img[imghelix+1]) :
            interstarts = []
            interends = []
            for dom in range(1, len(img[imghelix])):
                if (bool(img[imghelix][dom] and img[imghelix+1][dom])) and (not bool(img[imghelix][dom-1] and img[imghelix+1][dom-1])):
                    interstarts.append(dom)
                if (not bool(img[imghelix][dom] and img[imghelix+1][dom])) and (bool(img[imghelix][dom-1] and img[imghelix+1][dom-1])):
                    interends.append(dom)
            
            interstarts = [x*block for x in interstarts]
            interends = [x*block for x in interends]

        
        for i in range(len(interstarts)):

            #finds the scaffold crossover offset position 'nickoff', for the square lattice, 
            #closest to the center of the intersection of helices that are crossing over
            closestoff = int((interstarts[i]+interends[i])/2)-2
            closerem = (closestoff)%32

            if (closerem<10) :
                nickoff = 3 + 32*int(closestoff/32)
                if nickoff <= interstarts[i]:
                    nickoff = 3 + 32*int(closestoff/32)+10
                elif nickoff >= interends[i]:
                    nickoff = 3 + 32*int(closestoff/32)-11
            elif (closerem>10) and (closerem <21) :
                nickoff = 3 + 32*int(closestoff/32)+10
                if nickoff <= interstarts[i]:
                    nickoff = 3 + 32*int(closestoff/32)+21
                elif nickoff >= interends[i]:
                    nickoff = 3 + 32*int(closestoff/32)
            elif (closerem>21) and (closerem <32) :
                nickoff = 3 + 32*int(closestoff/32)+21
                if nickoff <= interstarts[i]:
                    nickoff = 3 + 32*int(closestoff/32)+32
                elif nickoff >= interends[i]:
                    nickoff = 3 + 32*int(closestoff/32)+10               
            else :
                nickoff = 3 + 32*int(closestoff/32)+ closerem
            
   
            design.add_nick(helix=helix, offset= nickoff, forward=helix % 2 == 0)

            design.add_nick(helix=helix+1, offset= nickoff, forward=helix % 2 == 1)
            crossovers.append(
                sc.Crossover(helix=helix, helix2=helix + 1, offset=nickoff, forward=False))

    design.add_nick(helix=helix, offset=nickoff, forward=helix % 2 == 0)
    design.add_crossovers(crossovers)

def add_scaffold_crossovers(design: sc.Design):
    crossovers = []

    # scaffold edges
    for helix in range(len(img)):
        if (1 in img[helix]) :
            for dom in range(len(img[helix])):
                if (img[helix][dom]==1) and (img[helix][dom-1]==0) :
                    start = block*dom
                if (img[helix][dom]==0) and (img[helix][dom-1]==1) :
                    end = block*dom

                    crossovers.append(sc.Crossover(helix=2*helix, helix2=2*helix + 1, offset=start, forward=True, half=True))
                    crossovers.append(sc.Crossover(helix=2*helix, helix2=2*helix + 1, offset=end-1, forward=True, half=True))

    design.add_crossovers(crossovers)


if __name__ == '__main__':
    design = create_design()
    design.write_scadnano_file()
    design.export_cadnano_v2()
