#### <span style="color:red"> Probabilistic model </span>

**This classifier classifies an input image either red, yellow or green based on *probabilities*.**

For a single input image, this classifier calculates **3 probabilities**:
1. Probability of image being red
2. Probability of image being yellow
3. Probability of image being green

And propobilities are calculated by,
- $Probability\ of\ image\ being\ red    = \dfrac {strength_{red}}    {strength_{red} + strength_{yellow} + strength_{green}}$
- $Probability\ of\ image\ being\ yellow = \dfrac {strength_{yellow}} {strength_{red} + strength_{yellow} + strength_{green}}$
- $Probability\ of\ image\ being\ green  = \dfrac {strength_{green}}  {strength_{red} + strength_{yellow} + strength_{green}}$

  where,  
  - $strength_{red}    = \mu_{saturation_{red}}^a    \times \mu_{brightness_{red}}^b$
  - $strength_{yellow} = \mu_{saturation_{yellow}}^a \times \mu_{brightness_{yellow}}^b$
  - $strength_{green}  = \mu_{saturation_{green}}^a  \times \mu_{brightness_{green}}^b$
  
  and,  
    - $\mu_{saturation_{red}}$   : mean saturation of model's red    hues extracted from the light located in model's red    light region
    - $\mu_{brightness_{red}}$   : mean brightness of model's red    hues extracted from the light located in model's red    light region
    - $\mu_{saturation_{yellow}}$: mean saturation of model's yellow hues extracted from the light located in model's yellow light region
    - $\mu_{brightness_{yellow}}$: mean brightness of model's yellow hues extracted from the light located in model's yellow light region
    - $\mu_{saturation_{green}}$ : mean saturation of model's green  hues extracted from the light located in model's green  light region
    - $\mu_{brightness_{green}}$ : mean brightness of model's green  hues extracted from the light located in model's green  light region
    - $a$ & $b$     : model's parameters

#### <span style="color:skyblue"> Hues extraction </span>

For a single image, the classifier extracts the following hues from the 3 regions:
1. Model's red    hues extracted from the light located in model's red    light region
2. Model's yellow hues extracted from the light located in model's yellow light region
3. Model's green  hues extracted from the light located in model's green  light region
