function [ys,ya,inv]=testy(Cl,Cs,Cp,fd,Cp_step)
 
        
        inv=0;

        x1=(abs(Cp^2-Cs^2)^0.5*pi*fd)/(Cp*Cs);    % numerator  
        x2=(abs(Cp^2-Cl^2)^0.5*pi*fd)/(Cp*Cl);    % denominator     
        y1=4*Cs^3*abs((Cp^2-Cl^2)*(Cp^2-Cs^2))^0.5;  % numerator for symmetric
        y2=Cl*(2*Cs^2-Cp^2)^2; % denominator for asymmetric       
        
        x1_temp=(abs((Cp+Cp_step)^2-Cs^2)^0.5*pi*fd)/((Cp+Cp_step)*Cs);
        x2_temp=(abs((Cp+Cp_step)^2-Cl^2)^0.5*pi*fd)/((Cp+Cp_step)*Cl);
        
        
        if Cp<=Cs
        ys=tanh(x1)*y2-tanh(x2)*y1; % for symmetric
        ya=tanh(x1)*y1-tanh(x2)*y2; % for asymmetric 
        elseif (Cp<=Cl)&&(Cp>Cs)
        ys=tan(x1)*y2-tanh(x2)*y1;  % for symmetric
        ya=tan(x1)*y1+tanh(x2)*y2;  % for asymmetric       
        else
        ys=tan(x1)*y2+tan(x2)*y1;   % for symmetric
        ya=tan(x1)*y1+tan(x2)*y2;   % for asymmetric 
        end   

        if (tan(x1)*tan(x1_temp)<0)||(tan(x2)*tan(x2_temp)<0)
            inv=1;
        else
        end
        
end
  