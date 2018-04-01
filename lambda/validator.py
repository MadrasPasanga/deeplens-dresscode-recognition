class AttireValidator():
 
   def default_action(frame, label, color, xmin, xmax, ymin, ymax):
       print(frame, label, color, xmin, ymin, xmax, ymax)
 
 
   def __init__(self, frame, action=default_action):
        self.objects = []
        self.personList = []
        self.action = action
        self.frame = frame
 
        # valid and invalid Tupe values
        self.validTuple = ['Blazer','Jeans']
        self.invalidTuple = ['Jeans','Tee']
 
        # categorizes the person based on dress code as valid, invalid and manual check
        self.validPersonMap = {}
        self.invalidPersonMap = {}
        self.manualCheckPersonMap = {}
 
 
   def addRecognitionObject(self, value):
        self.objects.append(value)
 
   def processObjects(self):
       # Find all persons in the frame
       for x in self.objects:
           if x.label == "Person":
              self.personList.append(x)
 
       for x in self.personList:
           	  tempDressList = []
           	  tempDressListLabel = []
           	  for y in self.objects:
                      # Not relying on Face recognition as the probability while moving is less. TODO: Improve training dataset
           	      if y.label == "Person" or y.label == "Face":
           	         continue
                      # Adding/Subtracting buffer to attire dimensions to ensure it falls within Person's dimensions
           	      if (y.xmin+15) >= x.xmin and (y.xmax-15) <= x.xmax and (y.ymin+15) >= x.ymin and (y.ymax-15) <= x.ymax:
           	          tempDressList.append(y)
           	          tempDressListLabel.append(y.label)
           	  tempDressListLabelTuple = sorted(tuple(tempDressListLabel))
           	  if tempDressListLabelTuple == self.invalidTuple:
           	     self.invalidPersonMap[x] = tempDressList
           	  elif tempDressListLabelTuple == self.validTuple:
           	     self.validPersonMap[x] = tempDressList
           	  else:
           	     self.manualCheckPersonMap[x] = tempDressList
 
       if len(self.validPersonMap) > 0:
           for person in self.validPersonMap.keys() :
               self.action(self.frame, "Allow", (0,255,0), person.xmin, person.xmax, person.ymin, person.ymax)
 
       if len(self.invalidPersonMap) > 0:
           for person in self.invalidPersonMap.keys() :
               self.action(self.frame, "Deny", (0,0,255), person.xmin, person.xmax, person.ymin, person.ymax)

       if len(self.manualCheckPersonMap) > 0:
           for person in self.manualCheckPersonMap.keys() :
               self.action(self.frame, "Manual Check", (255,0,0), person.xmin, person.xmax, person.ymin, person.ymax)
 
   def isPersonInFrame(self):
       if len(self.personList) > 0:
          return True
       return False   

   def isFrameValid(self):
       if len(self.validPersonMap) > 0 and len(self.invalidPersonMap) == 0 and len(self.manualCheckPersonMap) == 0:
          return True
       return False
 
   def isFrameInvalid(self):
       if len(self.invalidPersonMap) > 0 and len(self.manualCheckPersonMap) == 0:
          return True
       return False
 
 
   def doesFrameNeedManualCheck(self):
       if len(self.manualCheckPersonMap) > 0:
          return True
       return False
