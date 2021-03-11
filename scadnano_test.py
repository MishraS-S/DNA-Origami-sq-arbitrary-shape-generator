# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:15:57 2020

@author: Shubham
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
import scadnano as sc
import trimesh
from scipy.optimize import minimize_scalar
from scipy.ndimage import label, generate_binary_structure


## Scadnano oddities to keep in mind
# Crossover position is indicated by its topright offset position (of the four offset positions)



#To-do
#0. DONE - Fix staples being too short - merge short ones with adjacent staples? - Check if this is only on the perimeter?
#0.5. Extremities too unstable - may be reinforce by having crossovers near edges and allow really short staples - OR truncate the scaffold to have edges only at crossover positions - might have to shift the whole design to optimize efficient boundaries
#1. Optimize rasterization of face (max fraction of area covered among all the rotations may be?)
#2. Scaffold routing through all faces - polyhedron net algorithm? Hamiltonian path of the dual
# polyhedron of our input shape (in other words the face graph) is the answer but that's not always findable 
#- settle with brute force backtracking algo?
#3. Modifying vertices (rotating faces to XY plane) in-place doesn't work, because flattening one face
#applies a non-affine transformation to all adjacent faces - therefore probably need to make my own
#polygon object preserving vertex, edge and face information (make sure no difference between face and facet
#in this new object)
#4. Alternate staple patter - staple crossovers where scaffold crossovers for the strand pair right above

#print('h='+str(helix)+' o='+str(offset))


scaflen = 8064 #scaffold length to be used
block = 14 #controls effective "pixel" width, height is fixed at 2 helices for now

def PolyArea(x,y): #Area of polygon coordinates given by x,y
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def strlen (img): #Returns length of scaffold used in origami made from img (boolean image)
    count = 0
    for x in np.nditer(img):
        if x == 1:
            count+=1
    return count*2*block

def posdiff (big,small): #Returns some huge number if small<big, since used in minimization optimization
    if big>=small:
        return big-small
    else:
        return 1000000

input_type = 'S'

if input_type=='P':

    #Polygon input
    #------
    
    r = np.array([35,10,35,60])
    c = np.array([8, 50,25,50])
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
    print(strlen(img))
    
    # #Plot shape of design
    # fig = plt.figure()
    # plt.plot(rp,cp)
    # # plt.axis('square')
    # plt.axis('equal')
    # plt.style.use('dark_background')
    # plt.show()
    
    #---------

elif input_type=='S':

    ##SVG File input
    #------
    
    shape = trimesh.load(file_obj=r"C:\Users\Shubham\Desktop\bat1.svg", file_type='svg')
    
    #length difference between length of scaffold strand used and scaffold in origami
    #made from rasterization of svg image shape, with resolution reso
    def lendiff(reso): 
        img = shape.rasterize(reso,(0,0))
        img = np.asarray(img)
        return posdiff(scaflen, strlen(img))
    maxres = minimize_scalar(lendiff,bounds=(0, 100), method='bounded')
    
    
    img = shape.rasterize(maxres.x,(0,0))
    
    #Manual resolution setting
    # img = shape.rasterize(20,(0,0))
    
    img = np.asarray(img)
    print(strlen(img))
    print(maxres.x)
    print(posdiff(scaflen, strlen(img)))
    
    maxhelix = 2*len(img) - 1
    cmax = len(img[0])+2
    img = np.insert(img, 0, values=False, axis=1)
    img = np.insert(img, 0, values=False, axis=0)
    # plt.imshow(img)
    
    #---------



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


#Want to detect disconnected background regions since otherwise holes are not detected
im_inv = 1-img 
labeled_array, num_features = label(im_inv)
if num_features != 1:
    raise Exception("Non-contiguous regions in image!")


def create_design():
    design = precursor_scaffolds()
    add_scaffold_nicks(design)
    add_scaffold_crossovers(design)
    design.strands[0].set_scaffold()
    precursor_staples(design)
    add_staple_nicks(design)
    # add_staple_nicks_LD(design)
    add_deletions(design)
    return design

def domain_end(design: sc.Design, helix, offset): #True if it is a domain end or if nothing exists
    if len(design.domains_at(helix, offset))==2:
        if design.domains_at(helix, offset)[0].start == offset or \
           design.domains_at(helix, offset)[0].end == offset+1 or \
           design.domains_at(helix, offset)[1].start == offset or \
           design.domains_at(helix, offset)[1].end == offset+1 :
               return True
        else :
             return False
    elif len(design.domains_at(helix, offset))==1:
        if design.domains_at(helix, offset)[0].start == offset or \
           design.domains_at(helix, offset)[0].end == offset+1 : 
               return True
        else:
            return False
    else:
        return True #since we don't want crossovers where nothing exists


def long_stap_cond(design, helix, offset): #check if nicking at offset will lead to staples with minimum length "gap"
    gap = 4
    if (design.domains_at(helix, offset-gap)) and \
       (design.domains_at(helix, offset+gap)):
        return True
    else:
        return False
       

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


def cross_conds(design, helix, offset): #conditions for making crossover at offset
    if 0<=(offset)<=block*cmax and \
       (design.domains_at(helix, offset-4)) and \
       (design.domains_at(helix, offset+4)) and \
       (design.domains_at(helix+1, offset)) and \
       (design.domains_at(helix+1, offset-1)) and \
       not domain_end(design, helix, offset) and \
       not domain_end(design, helix+1, offset):  
        return True
    else :
        return False



def add_staple_nicks(design: sc.Design):
    
    crossovers = []
    midgap = 3
    state = 0
    every = -1 #skip every xth crossover
    
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

                
                if (((offset-3)%32==0) or \
                      ((offset-3)%32==10) or \
                      ((offset-3)%32==21)) and \
                      (design.domains_at(helix, offset+midgap)):
                         
                    state = state+1 
                    if not every==-1:
                        if state%every == 0:
                            # print(str(helix)+' '+str(offset))
                            continue
                    if domain_end(design, helix, offset+midgap) : #no nick if domain end
                        continue
                    #Prevent very short staples
                    if long_stap_cond(design, helix, offset+midgap)==True:
                        design.add_nick(helix=helix, offset=offset+midgap, forward=helix % 2 == 1)
                    if cross_conds(design, helix, offset):              
                        crossovers.append(sc.Crossover(helix=helix, helix2=helix + 1, \
                                                    offset=offset, forward=helix % 2 == 1))
                    elif helix==2:
                        print("not here "+str(offset))

        else :
            for offset in range(block*cmax):
                
                if (((offset-8)%32==0) or \
                      ((offset-8)%32==11) or \
                      ((offset-8)%32==21)) and \
                      (design.domains_at(helix, offset+midgap)):

                    state = state+1 
                    if not every==-1:
                        if state%every == 0:
                            # print(str(helix)+' '+str(offset))
                            continue                         
                    if domain_end(design, helix, offset+midgap) : #no nick if domain end
                        continue
                    # #Prevent very short staples
                    # if long_stap_cond(design, helix, offset+midgap)==True:
                    #     design.add_nick(helix=helix, offset=offset+midgap, forward=helix % 2 == 1)
                   
                    if cross_conds(design, helix, offset):              
                        crossovers.append(sc.Crossover(helix=helix, helix2=helix + 1, \
                                                    offset=offset, forward=helix % 2 == 1))

        
    design.add_crossovers(crossovers)




        
def add_staple_nicks_LD(design: sc.Design):
    
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
                if (bool(img[imghelix][dom] and img[imghelix+1][dom])) and \
                   (not bool(img[imghelix][dom-1] and img[imghelix+1][dom-1])):
                    interstarts.append(dom)
                if (not bool(img[imghelix][dom] and img[imghelix+1][dom])) and \
                   (bool(img[imghelix][dom-1] and img[imghelix+1][dom-1])):
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

def add_deletions(design: sc.Design):
    
    for helix in range(3, maxhelix): 
        if not design.strands_starting_on_helix(helix):
            pass
        else :
            hel1 = helix #first starting helix
            break
    
            
    for helix in range(hel1, maxhelix, 2):
        for offset in range(block*cmax):
            if design.domains_at(helix, offset) and offset%48==0:
                design.add_deletion(helix, offset)


if __name__ == '__main__':
    design = create_design()
    design.write_scadnano_file()
    design.export_cadnano_v2()
    
    #Plot staple length distribution
    strand_lengths = []
    for strand in design.strands:
        if not strand.dna_length() > 2000: 
            strand_lengths.append(strand.dna_length())
    hist = np.histogram(strand_lengths, bins=200, density=False)
    plt.plot(hist[1][:-1], hist[0])
    plt.style.use('dark_background')
    plt.show()
