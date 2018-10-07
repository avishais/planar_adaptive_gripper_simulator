
import numpy as np

def get_matrices(q, l, m, mo, Io, ro, c, k, z, a):

    w1 = 0; w2 = 0
    l1 = l[0]; l2 = l[1]
    h1 = c[0]; h2 = c[1]
    k1 = k[0]; k2 = k[1]
    z1 = z[0]; z2 = z[1]; z3 = z[2]; z4 = z[3]
    al = a[0]; ar = a[1]

    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]
    q7 = q[6]
    dq1 = q[7]
    dq2 = q[8]
    dq3 = q[9]
    dq4 = q[10]
    dq5 = q[11]
    dq6 = q[12]
    dq7 = q[13]

    DDt = Dt(q1, q2, q3, q4, w1, w2, l1, l2, ro, Io, mo, m)
    CCt = Ct(q1, q2, q3, q4, dq1, dq2, dq3, dq4, l1, l2, h1, h2, w1, w2, m, ro)
    GGt = Gt(q1, q2, q3, q4, w1, w2, z1, z2, z3, z4, k1, k2, l1, l2, ro) 
    FFt = Ft(q1, q2, q3, q4, l1, l2, w1, w2, ro)   
    eeee = ee(q1, q2, q3, q4, al, ar, l1, l2)
    JJee = Jee(q1, q2, q3, q4, l1, l2)
    ddJG = dJG(q1, q2, q3, q4, dq1, dq2, dq3, dq4, l1, l2, w1, w2, ro)
    GGmap = Gmap(w1, w2, ro)

    return DDt, CCt, GGt, FFt, GGmap, eeee, JJee, ddJG


def Gt(q1, q2, q3, q4, w1, w2, z1, z2, z3, z4, k1, k2, l1, l2, ro):

    t5 = np.pi*(1.0/2.0);
    t6 = (q1);
    t7 = t5+t6;
    t2 = np.cos(t7);
    t3 = (q2);
    t4 = np.sin(t3);
    t8 = np.sin(t7);
    t9 = (l1);
    t10 = (l2);
    t11 = t2**2;
    t12 = t4*t9*t10*t11;
    t13 = t8**2;
    t14 = t4*t9*t10*t13;
    t15 = t12+t14;
    t16 = 1.0/t15;
    t17 = np.cos(t3);
    t21 = (q3);
    t22 = t5+t21;
    t18 = np.cos(t22);
    t19 = (q4);
    t20 = np.sin(t19);
    t23 = np.sin(t22);
    t24 = t18**2;
    t25 = t9*t10*t20*t24;
    t26 = t23**2;
    t27 = t9*t10*t20*t26;
    t28 = t25+t27;
    t29 = 1.0/t28;
    t30 = np.cos(t19);
    t31 = np.cos(w1);
    t32 = np.sin(w1);
    t33 = t4*t9*t11;
    t34 = t4*t9*t13;
    t35 = t33+t34;
    t36 = 1.0/t35;
    t37 = np.cos(w2);
    t38 = np.sin(w2);
    t39 = t9*t20*t24;
    t40 = t9*t20*t26;
    t41 = t39+t40;
    t42 = 1.0/t41;
    t43 = k2*q2;
    t76 = k2*z2;
    t44 = t43-t76;
    t45 = t8*t9;
    t46 = t8*t10*t17;
    t47 = t2*t4*t10;
    t48 = t45+t46+t47;
    t49 = t2*t9;
    t50 = t2*t10*t17;
    t75 = t4*t8*t10;
    t51 = t49+t50-t75;
    t52 = k2*q4;
    t78 = k2*z4;
    t53 = t52-t78;
    t54 = t9*t23;
    t55 = t10*t23*t30;
    t56 = t10*t18*t20;
    t57 = t54+t55+t56;
    t58 = t9*t18;
    t59 = t10*t18*t30;
    t77 = t10*t20*t23;
    t60 = t58+t59-t77;
    t61 = t8*t17;
    t62 = t2*t4;
    t63 = t61+t62;
    t64 = t4*t8;
    t79 = t2*t17;
    t65 = t64-t79;
    t66 = k1*q1;
    t80 = k1*z1;
    t67 = t66-t80;
    t68 = t23*t30;
    t69 = t18*t20;
    t70 = t68+t69;
    t71 = t20*t23;
    t81 = t18*t30;
    t72 = t71-t81;
    t73 = k1*q3;
    t82 = k1*z3;
    t74 = t73-t82;
    Gt = [t44*(t16*t32*t48-t16*t31*t51)+t53*(t29*t38*t57-t29*t37*t60)-t67*(t32*t36*t63+t31*t36*t65)-t74*(t38*t42*t70+t37*t42*t72),-t44*(t16*t31*t48+t16*t32*t51)-t53*(t29*t37*t57+t29*t38*t60)+t67*(t31*t36*t63-t32*t36*t65)+t74*(t37*t42*t70-t38*t42*t72),t44*(ro*t16*t31*t48+ro*t16*t32*t51)-t53*(ro*t29*t37*t57+ro*t29*t38*t60)-t67*(ro*t31*t36*t63-ro*t32*t36*t65)+t74*(ro*t37*t42*t70-ro*t38*t42*t72)];

    return np.array(Gt).reshape(3,1)

def Dt(q1, q2, q3, q4, w1, w2, l1, l2, ro, Io, mo, m):

    t2 = (q2);
    t3 = np.pi*(1.0/2.0);
    t4 = (q1);
    t5 = t3+t4;
    t6 = np.cos(t5);
    t7 = np.sin(t2);
    t8 = np.sin(t5);
    t9 = (l1);
    t10 = np.cos(t2);
    t11 = t6**2;
    t12 = t7*t9*t11;
    t13 = t8**2;
    t14 = t7*t9*t13;
    t15 = t12+t14;
    t16 = 1.0/t15;
    t17 = l2**2;
    t18 = np.cos(w1);
    t19 = (l2);
    t20 = np.sin(w1);
    t21 = t7*t9*t11*t19;
    t22 = t7*t9*t13*t19;
    t23 = t21+t22;
    t24 = 1.0/t23;
    t27 = q1+t3;
    t25 = np.cos(t27);
    t26 = np.sin(q2);
    t28 = np.sin(t27);
    t29 = np.cos(q2);
    t30 = t28**2;
    t31 = t25**2;
    t32 = m*t17*(5.0/4.0);
    t33 = l1*l2*m*t29*(1.0/2.0);
    t34 = t32+t33;
    t35 = t6*t9;
    t36 = t6*t10*t19;
    t98 = t7*t8*t19;
    t37 = t35+t36-t98;
    t38 = t8*t9;
    t39 = t8*t10*t19;
    t40 = t6*t7*t19;
    t41 = t38+t39+t40;
    t99 = t18*t24*t37;
    t100 = t20*t24*t41;
    t42 = t99-t100;
    t43 = t7*t8;
    t95 = t6*t10;
    t44 = t43-t95;
    t45 = t16*t18*t44;
    t46 = t8*t10;
    t47 = t6*t7;
    t48 = t46+t47;
    t49 = t16*t20*t48;
    t50 = (q4);
    t51 = (q3);
    t52 = t3+t51;
    t53 = np.cos(t52);
    t54 = np.sin(t50);
    t55 = np.sin(t52);
    t56 = np.cos(t50);
    t57 = t53**2;
    t58 = t9*t54*t57;
    t59 = t55**2;
    t60 = t9*t54*t59;
    t61 = t58+t60;
    t62 = 1.0/t61;
    t63 = np.cos(w2);
    t64 = np.sin(w2);
    t65 = t9*t19*t54*t57;
    t66 = t9*t19*t54*t59;
    t67 = t65+t66;
    t68 = 1.0/t67;
    t71 = q3+t3;
    t69 = np.cos(t71);
    t70 = np.sin(q4);
    t72 = np.sin(t71);
    t73 = np.cos(q4);
    t74 = t72**2;
    t75 = t69**2;
    t76 = l1*l2*m*t73*(1.0/2.0);
    t77 = t32+t76;
    t78 = t9*t53;
    t79 = t19*t53*t56;
    t119 = t19*t54*t55;
    t80 = t78+t79-t119;
    t81 = t9*t55;
    t82 = t19*t55*t56;
    t83 = t19*t53*t54;
    t84 = t81+t82+t83;
    t120 = t63*t68*t80;
    t121 = t64*t68*t84;
    t85 = t120-t121;
    t86 = t53*t56;
    t115 = t54*t55;
    t87 = t86-t115;
    t88 = t55*t56;
    t89 = t53*t54;
    t90 = t88+t89;
    t116 = t62*t63*t87;
    t117 = t62*t64*t90;
    t91 = t116-t117;
    t92 = l1**2;
    t93 = m*t92*(9.0/4.0);
    t94 = (w1);
    t96 = t45+t49;
    t97 = t34*t96;
    t101 = m*t17*t42*(5.0/4.0);
    t102 = l1*l2*t26*t31;
    t103 = l1*l2*t26*t30;
    t104 = t102+t103;
    t105 = 1.0/t104;
    t106 = t97+t101;
    t107 = l1*t26*t30;
    t108 = l1*t26*t31;
    t109 = t107+t108;
    t110 = 1.0/t109;
    t111 = t34*t42;
    t112 = l1*l2*m*t29;
    t113 = t32+t93+t112;
    t114 = (w2);
    t118 = t77*t91;
    t152 = m*t17*t85*(5.0/4.0);
    t122 = t118-t152;
    t123 = l1*l2*t70*t75;
    t124 = l1*l2*t70*t74;
    t125 = t123+t124;
    t126 = 1.0/t125;
    t127 = l1*t70*t74;
    t128 = l1*t70*t75;
    t129 = t127+t128;
    t130 = 1.0/t129;
    t131 = l1*l2*m*t73;
    t132 = t32+t93+t131;
    t161 = t77*t85;
    t162 = t91*t132;
    t133 = t161-t162;
    t134 = np.cos(t94);
    t135 = l1*t28;
    t136 = l2*t28*t29;
    t137 = l2*t25*t26;
    t138 = t135+t136+t137;
    t139 = t105*t106*t138;
    t140 = t28*t29;
    t141 = t25*t26;
    t142 = t140+t141;
    t143 = t113*(t45+t49);
    t144 = t111+t143;
    t145 = np.sin(t94);
    t146 = l1*t25;
    t147 = l2*t25*t29;
    t176 = l2*t26*t28;
    t148 = t146+t147-t176;
    t149 = t25*t29;
    t178 = t26*t28;
    t150 = t149-t178;
    t151 = np.cos(t114);
    t153 = l1*t72;
    t154 = l2*t72*t73;
    t155 = l2*t69*t70;
    t156 = t153+t154+t155;
    t157 = t122*t126*t156;
    t158 = t72*t73;
    t159 = t69*t70;
    t160 = t158+t159;
    t163 = t130*t133*t160;
    t164 = t157+t163;
    t165 = np.sin(t114);
    t166 = l1*t69;
    t167 = l2*t69*t73;
    t180 = l2*t70*t72;
    t168 = t166+t167-t180;
    t169 = t122*t126*t168;
    t170 = t69*t73;
    t181 = t70*t72;
    t171 = t170-t181;
    t172 = t130*t133*t171;
    t173 = t169+t172;
    t174 = t139-t110*t142*t144;
    t175 = (ro);
    t177 = t105*t106*t148;
    t179 = t177-t110*t144*t150;
    t182 = t18*t24*t41;
    t183 = t20*t24*t37;
    t184 = t182+t183;
    t185 = t16*t20*t44;
    t186 = t63*t68*t84;
    t187 = t64*t68*t80;
    t188 = t186+t187;
    t189 = t62*t63*t90;
    t190 = t62*t64*t87;
    t191 = t189+t190;
    t197 = t16*t18*t48;
    t192 = t185-t197;
    t193 = t34*t192;
    t194 = m*t17*t184*(5.0/4.0);
    t195 = t193+t194;
    t196 = t34*t184;
    t198 = t113*t192;
    t199 = t196+t198;
    t200 = t77*t191;
    t208 = m*t17*t188*(5.0/4.0);
    t201 = t200-t208;
    t202 = t77*t188;
    t210 = t132*t191;
    t203 = t202-t210;
    t204 = t105*t138*t195;
    t205 = t105*t148*t195;
    t206 = t113*(t185-t197);
    t207 = t196+t206;
    t209 = t126*t156*t201;
    t211 = t130*t160*t203;
    t212 = t209+t211;
    t213 = t126*t168*t201;
    t214 = t130*t171*t203;
    t215 = t213+t214;
    t216 = t204-t110*t142*t207;
    t217 = t205-t110*t150*t207;
    t218 = ro*t18*t24*t41;
    t219 = ro*t20*t24*t37;
    t220 = t218+t219;
    t221 = ro*t16*t20*t44;
    t229 = ro*t16*t18*t48;
    t222 = t221-t229;
    t223 = ro*t63*t68*t84;
    t224 = ro*t64*t68*t80;
    t225 = t223+t224;
    t226 = ro*t62*t64*t87;
    t227 = ro*t62*t63*t90;
    t228 = t226+t227;
    t230 = t34*t222;
    t231 = m*t17*t220*(5.0/4.0);
    t232 = t230+t231;
    t233 = t34*t220;
    t234 = t113*t222;
    t235 = t233+t234;
    t236 = t77*t228;
    t244 = m*t17*t225*(5.0/4.0);
    t237 = t236-t244;
    t238 = t77*t225;
    t246 = t132*t228;
    t239 = t238-t246;
    t240 = t105*t138*t232;
    t241 = t240-t110*t142*t235;
    t242 = t105*t148*t232;
    t243 = t242-t110*t150*t235;
    t245 = t126*t156*t237;
    t247 = t130*t160*t239;
    t248 = t245+t247;
    t249 = t126*t168*t237;
    t250 = t130*t171*t239;
    t251 = t249+t250;
    t252 = t113*(t221-t229);
    t253 = t233+t252;
    Dt = np.array([[mo-t145*(t139-t110*t142*(t111+t96*t113))+t134*t179-t151*t173+t164*t165,-t151*t215+t165*t212+t134*(t205-t110*t150*t199)-t145*(t204-t110*t142*t199),-t134*t243+t145*t241-t151*t251+t165*t248],[t134*t174-t151*t164+t145*t179-t165*t173,mo+t134*t216+t145*t217-t151*t212-t165*t215,-t134*t241-t145*t243-t151*t248-t165*t251],[-t134*t174*t175-t151*t164*t175-t145*t175*t179-t165*t173*t175,-t134*t175*t216-t145*t175*t217-t151*t175*t212-t165*t175*t215,Io-t151*t175*t248-t165*t175*t251+t134*t175*(t240-t110*t142*t253)+t145*t175*(t242-t110*t150*t253)]]).reshape(3,3)

    return np.transpose(Dt)

def Ct(q1, q2, q3, q4, dq1, dq2, dq3, dq4, l1, l2, h1, h2, w1, w2, m, ro):

    t5 = np.pi*(1.0/2.0);
    t6 = (q1);
    t7 = t5+t6;
    t2 = np.cos(t7);
    t3 = (q2);
    t4 = np.sin(t3);
    t8 = np.sin(t7);
    t9 = (l1);
    t10 = (l2);
    t11 = t2**2;
    t12 = t4*t9*t10*t11;
    t13 = t8**2;
    t14 = t4*t9*t10*t13;
    t15 = t12+t14;
    t16 = 1.0/t15;
    t17 = np.cos(t3);
    t19 = q1+t5;
    t18 = np.sin(t19);
    t20 = np.cos(t19);
    t21 = np.sin(q2);
    t22 = np.cos(q2);
    t23 = (w1);
    t24 = t18**2;
    t25 = l1*t21*t24;
    t26 = t20**2;
    t27 = l1*t21*t26;
    t28 = t25+t27;
    t29 = 1.0/t28;
    t30 = np.sin(t23);
    t31 = np.cos(t23);
    t32 = 1.0/t28**2;
    t33 = dq2*l1*t22*t26;
    t34 = dq2*l1*t22*t24;
    t35 = t33+t34;
    t36 = t18*t21;
    t37 = l1*l2*t21*t26;
    t38 = l1*l2*t21*t24;
    t39 = t37+t38;
    t40 = 1.0/t39;
    t201 = dq1*l1*l2*m*t21*(1.0/2.0);
    t41 = h2-t201;
    t42 = t18*t22;
    t43 = t20*t21;
    t44 = t42+t43;
    t45 = l2**2;
    t46 = l1*t20;
    t47 = l2*t20*t22;
    t118 = l2*t18*t21;
    t48 = t46+t47-t118;
    t49 = 1.0/t39**2;
    t50 = dq2*l1*l2*t22*t26;
    t51 = dq2*l1*l2*t22*t24;
    t52 = t50+t51;
    t53 = l1*t18;
    t54 = l2*t18*t22;
    t55 = l2*t20*t21;
    t56 = t53+t54+t55;
    t117 = t20*t22;
    t57 = t36-t117;
    t61 = (q3);
    t62 = t5+t61;
    t58 = np.cos(t62);
    t59 = (q4);
    t60 = np.sin(t59);
    t63 = np.sin(t62);
    t64 = t58**2;
    t65 = t9*t10*t60*t64;
    t66 = t63**2;
    t67 = t9*t10*t60*t66;
    t68 = t65+t67;
    t69 = 1.0/t68;
    t70 = np.cos(t59);
    t71 = m*t45*(5.0/4.0);
    t73 = q3+t5;
    t72 = np.sin(t73);
    t74 = np.cos(t73);
    t75 = np.sin(q4);
    t76 = np.cos(q4);
    t77 = (w2);
    t78 = t72**2;
    t79 = l1*t75*t78;
    t80 = t74**2;
    t81 = l1*t75*t80;
    t82 = t79+t81;
    t83 = 1.0/t82;
    t84 = np.sin(t77);
    t85 = np.cos(t77);
    t86 = 1.0/t82**2;
    t87 = dq4*l1*t76*t80;
    t88 = dq4*l1*t76*t78;
    t89 = t87+t88;
    t90 = t72*t75;
    t91 = l1*l2*t75*t80;
    t92 = l1*l2*t75*t78;
    t93 = t91+t92;
    t94 = 1.0/t93;
    t217 = dq3*l1*l2*m*t75*(1.0/2.0);
    t95 = h2-t217;
    t96 = t72*t76;
    t97 = t74*t75;
    t98 = t96+t97;
    t99 = l1*t74;
    t100 = l2*t74*t76;
    t157 = l2*t72*t75;
    t101 = t99+t100-t157;
    t102 = 1.0/t93**2;
    t103 = dq4*l1*l2*t76*t80;
    t104 = dq4*l1*l2*t76*t78;
    t105 = t103+t104;
    t106 = l1*t72;
    t107 = l2*t72*t76;
    t108 = l2*t74*t75;
    t109 = t106+t107+t108;
    t156 = t74*t76;
    t110 = t90-t156;
    t111 = np.cos(w1);
    t112 = np.sin(w1);
    t113 = t4*t9*t11;
    t114 = t4*t9*t13;
    t115 = t113+t114;
    t116 = 1.0/t115;
    t119 = dq1*l1*l2*m*t21;
    t120 = dq2*l1*l2*m*t21*(1.0/2.0);
    t121 = t119+t120;
    t122 = dq1*t18*t22;
    t123 = dq1*t20*t21;
    t124 = dq2*t18*t22;
    t125 = dq2*t20*t21;
    t126 = t122+t123+t124+t125;
    t127 = dq1*t18*t21;
    t128 = dq2*t18*t21;
    t199 = dq1*t20*t22;
    t200 = dq2*t20*t22;
    t129 = t127+t128-t199-t200;
    t130 = t29*t30*t129;
    t131 = t30*t32*t35*t44;
    t132 = t31*t32*t35*t57;
    t133 = l1*l2*m*t22*(1.0/2.0);
    t134 = t71+t133;
    t135 = dq1*l1*t18;
    t136 = dq1*l2*t18*t22;
    t137 = dq1*l2*t20*t21;
    t138 = dq2*l2*t18*t22;
    t139 = dq2*l2*t20*t21;
    t140 = t135+t136+t137+t138+t139;
    t141 = t31*t40*t140;
    t142 = dq1*l1*t20;
    t143 = dq1*l2*t20*t22;
    t144 = dq2*l2*t20*t22;
    t205 = dq1*l2*t18*t21;
    t206 = dq2*l2*t18*t21;
    t145 = t142+t143+t144-t205-t206;
    t146 = t30*t40*t145;
    t147 = t31*t48*t49*t52;
    t305 = t30*t49*t52*t56;
    t148 = t141+t146+t147-t305;
    t302 = t29*t31*t126;
    t149 = t130+t131+t132-t302;
    t150 = np.cos(w2);
    t151 = np.sin(w2);
    t152 = t9*t60*t64;
    t153 = t9*t60*t66;
    t154 = t152+t153;
    t155 = 1.0/t154;
    t158 = dq3*l1*l2*m*t75;
    t159 = dq4*l1*l2*m*t75*(1.0/2.0);
    t160 = t158+t159;
    t161 = l1**2;
    t162 = m*t161*(9.0/4.0);
    t163 = dq3*t72*t76;
    t164 = dq3*t74*t75;
    t165 = dq4*t72*t76;
    t166 = dq4*t74*t75;
    t167 = t163+t164+t165+t166;
    t168 = dq3*t72*t75;
    t169 = dq4*t72*t75;
    t215 = dq3*t74*t76;
    t216 = dq4*t74*t76;
    t170 = t168+t169-t215-t216;
    t171 = t83*t84*t170;
    t172 = t84*t86*t89*t98;
    t173 = t85*t86*t89*t110;
    t174 = l1*l2*m*t76*(1.0/2.0);
    t175 = t71+t174;
    t176 = dq3*l1*t72;
    t177 = dq3*l2*t72*t76;
    t178 = dq3*l2*t74*t75;
    t179 = dq4*l2*t72*t76;
    t180 = dq4*l2*t74*t75;
    t181 = t176+t177+t178+t179+t180;
    t182 = t85*t94*t181;
    t183 = dq3*l1*t74;
    t184 = dq3*l2*t74*t76;
    t185 = dq4*l2*t74*t76;
    t221 = dq3*l2*t72*t75;
    t222 = dq4*l2*t72*t75;
    t186 = t183+t184+t185-t221-t222;
    t187 = t84*t94*t186;
    t188 = t85*t101*t102*t105;
    t310 = t84*t102*t105*t109;
    t189 = t182+t187+t188-t310;
    t307 = t83*t85*t167;
    t190 = t171+t172+t173-t307;
    t191 = t2*t9;
    t192 = t2*t10*t17;
    t269 = t4*t8*t10;
    t193 = t191+t192-t269;
    t194 = t8*t9;
    t195 = t8*t10*t17;
    t196 = t2*t4*t10;
    t197 = t194+t195+t196;
    t270 = t16*t111*t193;
    t271 = t16*t112*t197;
    t198 = t270-t271;
    t202 = t40*t41*t56;
    t273 = dq1*l1*l2*m*t21*t29*t44*(1.0/2.0);
    t203 = t202-t273;
    t204 = t40*t41*t48;
    t207 = t9*t58;
    t208 = t10*t58*t70;
    t276 = t10*t60*t63;
    t209 = t207+t208-t276;
    t210 = t9*t63;
    t211 = t10*t63*t70;
    t212 = t10*t58*t60;
    t213 = t210+t211+t212;
    t277 = t69*t150*t209;
    t278 = t69*t151*t213;
    t214 = t277-t278;
    t218 = t94*t95*t109;
    t279 = dq3*l1*l2*m*t75*t83*t98*(1.0/2.0);
    t219 = t218-t279;
    t220 = t94*t95*t101;
    t223 = t4*t8;
    t282 = t2*t17;
    t224 = t223-t282;
    t225 = t111*t116*t224;
    t226 = t8*t17;
    t227 = t2*t4;
    t228 = t226+t227;
    t229 = t112*t116*t228;
    t230 = h1*t29*t44;
    t231 = t40*t56*t121;
    t232 = t230+t231;
    t233 = h1*t29*t57;
    t291 = t40*t48*t121;
    t234 = t233-t291;
    t235 = l1*l2*m*t22;
    t236 = t71+t162+t235;
    t237 = t29*t31*(t127+t128-t199-t200);
    t238 = t29*t30*t126;
    t239 = t31*t32*t35*t44;
    t319 = t30*t32*t35*t57;
    t240 = t237+t238+t239-t319;
    t241 = t30*t40*t140;
    t242 = t31*t49*t52*t56;
    t243 = t30*t48*t49*t52;
    t323 = t31*t40*t145;
    t244 = t241+t242+t243-t323;
    t245 = t225+t229;
    t246 = t60*t63;
    t292 = t58*t70;
    t247 = t246-t292;
    t248 = t150*t155*t247;
    t249 = t63*t70;
    t250 = t58*t60;
    t251 = t249+t250;
    t252 = t151*t155*t251;
    t253 = h1*t83*t98;
    t254 = t94*t109*t160;
    t255 = t253+t254;
    t256 = h1*t83*t110;
    t301 = t94*t101*t160;
    t257 = t256-t301;
    t258 = l1*l2*m*t76;
    t259 = t71+t162+t258;
    t260 = t83*t85*(t168+t169-t215-t216);
    t261 = t83*t84*t167;
    t262 = t85*t86*t89*t98;
    t328 = t84*t86*t89*t110;
    t263 = t260+t261+t262-t328;
    t264 = t84*t94*t181;
    t265 = t85*t102*t105*t109;
    t266 = t84*t101*t102*t105;
    t332 = t85*t94*t186;
    t267 = t264+t265+t266-t332;
    t268 = t248+t252;
    t272 = (ro);
    t274 = dq1*l1*l2*m*t21*t29*(t36-t117)*(1.0/2.0);
    t275 = t204+t274;
    t280 = dq3*l1*l2*m*t75*t83*(t90-t156)*(1.0/2.0);
    t281 = t220+t280;
    t283 = t30*t40*t140*t272;
    t284 = t31*t49*t52*t56*t272;
    t285 = t30*t48*t49*t52*t272;
    t347 = t31*t40*t145*t272;
    t286 = t283+t284+t285-t347;
    t287 = t29*t30*t126*t272;
    t288 = t31*t32*t35*t44*t272;
    t289 = t29*t31*t129*t272;
    t344 = t30*t32*t35*t57*t272;
    t290 = t287+t288+t289-t344;
    t293 = t84*t94*t181*t272;
    t294 = t85*t102*t105*t109*t272;
    t295 = t84*t101*t102*t105*t272;
    t354 = t85*t94*t186*t272;
    t296 = t293+t294+t295-t354;
    t297 = t83*t85*t272*(t168+t169-t215-t216);
    t298 = t83*t84*t167*t272;
    t299 = t85*t86*t89*t98*t272;
    t350 = t84*t86*t89*t110*t272;
    t300 = t297+t298+t299-t350;
    t303 = t134*t149;
    t304 = t30*t203;
    t306 = m*t45*t148*(5.0/4.0);
    t308 = t175*t190;
    t309 = t84*t219;
    t311 = m*t45*t189*(5.0/4.0);
    t312 = t31*t234;
    t313 = t30*t232;
    t314 = t85*t257;
    t315 = t84*t255;
    t316 = t16*t111*t197;
    t317 = t16*t112*t193;
    t318 = t316+t317;
    t320 = t134*t240;
    t321 = t31*t203;
    t322 = t30*t275;
    t369 = m*t45*t244*(5.0/4.0);
    t324 = t320+t321+t322-t369;
    t325 = t69*t150*t213;
    t326 = t69*t151*t209;
    t327 = t325+t326;
    t329 = t175*t263;
    t330 = t85*t219;
    t331 = t84*t281;
    t373 = m*t45*t267*(5.0/4.0);
    t333 = t329+t330+t331-t373;
    t334 = t112*t116*t224;
    t335 = t31*t232;
    t336 = t134*t244;
    t375 = t30*t234;
    t376 = t236*t240;
    t337 = t335+t336-t375-t376;
    t356 = t111*t116*t228;
    t338 = t334-t356;
    t339 = t151*t155*t247;
    t340 = t85*t255;
    t341 = t175*t267;
    t378 = t84*t257;
    t379 = t259*t263;
    t342 = t340+t341-t378-t379;
    t360 = t150*t155*t251;
    t343 = t339-t360;
    t345 = t31*t203*t272;
    t346 = t30*t272*t275;
    t348 = t134*t290;
    t387 = m*t45*t286*(5.0/4.0);
    t349 = t345+t346+t348-t387;
    t351 = t175*t300;
    t352 = t85*t219*t272;
    t353 = t84*t272*t281;
    t388 = m*t45*t296*(5.0/4.0);
    t355 = t351+t352+t353-t388;
    t357 = t134*t286;
    t358 = t31*t232*t272;
    t381 = t236*t290;
    t382 = t30*t234*t272;
    t359 = t357+t358-t381-t382;
    t361 = t175*t296;
    t362 = t85*t255*t272;
    t385 = t259*t300;
    t386 = t84*t257*t272;
    t363 = t361+t362-t385-t386;
    t364 = t308+t309+t311-t85*t281;
    t365 = t312+t313-t134*t148-t149*t236;
    t366 = ro*t16*t111*t197;
    t367 = ro*t16*t112*t193;
    t368 = t366+t367;
    t370 = ro*t69*t150*t213;
    t371 = ro*t69*t151*t209;
    t372 = t370+t371;
    t374 = ro*t112*t116*t224;
    t377 = ro*t151*t155*t247;
    t384 = ro*t150*t155*t251;
    t380 = t377-t384;
    t383 = t374-ro*t111*t116*t228;
    Ct = np.array([[t245*t365-t198*(t303+t304+t306-t31*(t204+dq1*l1*l2*m*t21*t29*t57*(1.0/2.0)))-t214*(t308+t309+t311-t85*(t220+dq3*l1*l2*m*t75*t83*t110*(1.0/2.0)))+t268*(t314+t315-t175*t189-t190*t259),-t327*t364+t338*(t312+t313-t134*t148-t236*(t130+t131+t132-t302))+t343*(t314+t315-t175*t189-t259*(t171+t172+t173-t307))-t318*(t303+t304+t306-t31*t275),-t364*t372-t365*t383+t380*(t314+t315-t175*t189-t259*(t171+t172+t173-t307))+t368*(t303+t304+t306-t31*t275)],[t198*t324+t214*t333-t245*t337-t268*t342,t318*t324+t327*t333-t337*t338-t342*t343,-t324*t368+t333*t372+t337*t383-t342*t380],[t359*(t225+t229)-t198*t349+t214*t355-t268*t363,-t318*t349+t327*t355-t343*t363+t359*(t334-t356),t355*t372-t359*t383-t363*t380+t368*(t345+t346-t387+t134*(t287+t288-t344+t29*t31*t272*(t127+t128-t199-t200)))]]).reshape(3,3)

    return np.transpose(Ct)

def Ft(q1, q2, q3, q4, l1, l2, w1, w2, ro):

    t2 = (q2);
    t3 = np.pi*(1.0/2.0);
    t4 = (q1);
    t5 = t3+t4;
    t6 = np.cos(t5);
    t7 = np.sin(t2);
    t8 = np.sin(t5);
    t9 = (l1);
    t10 = np.cos(t2);
    t11 = t6**2;
    t12 = t7*t9*t11;
    t13 = t8**2;
    t14 = t7*t9*t13;
    t15 = t12+t14;
    t16 = 1.0/t15;
    t17 = np.cos(w1);
    t18 = (l2);
    t19 = np.sin(w1);
    t20 = t7*t9*t11*t18;
    t21 = t7*t9*t13*t18;
    t22 = t20+t21;
    t23 = 1.0/t22;
    t24 = (q4);
    t25 = (q3);
    t26 = t3+t25;
    t27 = np.cos(t26);
    t28 = np.sin(t24);
    t29 = np.sin(t26);
    t30 = np.cos(t24);
    t31 = t27**2;
    t32 = t9*t28*t31;
    t33 = t29**2;
    t34 = t9*t28*t33;
    t35 = t32+t34;
    t36 = 1.0/t35;
    t37 = np.cos(w2);
    t38 = np.sin(w2);
    t39 = t9*t18*t28*t31;
    t40 = t9*t18*t28*t33;
    t41 = t39+t40;
    t42 = 1.0/t41;
    t43 = t8*t10;
    t44 = t6*t7;
    t45 = t43+t44;
    t46 = t7*t8;
    t67 = t6*t10;
    t47 = t46-t67;
    t48 = t8*t9;
    t49 = t8*t10*t18;
    t50 = t6*t7*t18;
    t51 = t48+t49+t50;
    t52 = t6*t9;
    t53 = t6*t10*t18;
    t68 = t7*t8*t18;
    t54 = t52+t53-t68;
    t55 = t29*t30;
    t56 = t27*t28;
    t57 = t55+t56;
    t58 = t27*t30;
    t69 = t28*t29;
    t59 = t58-t69;
    t60 = t9*t29;
    t61 = t18*t29*t30;
    t62 = t18*t27*t28;
    t63 = t60+t61+t62;
    t64 = t9*t27;
    t65 = t18*t27*t30;
    t70 = t18*t28*t29;
    t66 = t64+t65-t70;
    Ft = np.array([[-t16*t17*t47-t16*t19*t45,t16*t17*t45-t16*t19*t47,-ro*t16*t17*t45+ro*t16*t19*(t46-t67)],[t19*t23*t51-t17*t23*t54,-t17*t23*t51-t19*t23*t54,ro*t17*t23*t51+ro*t19*t23*t54],[-t36*t38*t57+t36*t37*t59,t36*t37*t57+t36*t38*t59,ro*t36*t37*t57+ro*t36*t38*t59],[t38*t42*t63-t37*t42*t66,-t37*t42*t63-t38*t42*t66,-ro*t37*t42*t63-ro*t38*t42*t66]]).reshape(4,3)

    Ft = np.transpose(Ft)

    return Ft
    
def ee(q1, q2, q3, q4, al, ar, l1, l2):

    t2 = np.pi*(1.0/2.0);
    t3 = q1+t2;
    t4 = np.cos(t3);
    t5 = l1*t4;
    t6 = q3+t2;
    t7 = np.cos(t6);
    t8 = l1*t7;
    t9 = np.sin(t3);
    t10 = np.cos(q2);
    t11 = np.sin(q2);
    t12 = l1*t9;
    t13 = np.sin(t6);
    t14 = np.cos(q4);
    t15 = np.sin(q4);
    t16 = l1*t13;
    ee = np.array([[al+t5,t12],[al+t5+l2*(t4*t10-t9*t11),t12+l2*(t4*t11+t9*t10)],[ar+t8,t16],[ar+t8+l2*(t7*t14-t13*t15),t16+l2*(t7*t15+t13*t14)]]).reshape(4,2)
    ee = np.transpose(ee)

    return ee

def Jee(q1, q2, q3, q4, l1, l2):

    t2 = np.pi*(1.0/2.0);
    t3 = q1+t2;
    t4 = np.sin(t3);
    t5 = np.cos(q2);
    t6 = t4*t5;
    t7 = np.cos(t3);
    t8 = np.sin(q2);
    t9 = t7*t8;
    t10 = t6+t9;
    t11 = t4*t8;
    t12 = t11-t5*t7;
    t13 = q3+t2;
    t14 = np.sin(t13);
    t15 = np.cos(q4);
    t16 = t14*t15;
    t17 = np.cos(t13);
    t18 = np.sin(q4);
    t19 = t17*t18;
    t20 = t16+t19;
    t21 = t14*t18;
    t22 = t21-t15*t17;
    Jee = np.array([[float(-l1*t4-l2*t10),float(l1*t7-l2*t12),0.0,0.0],[float(-l2*t10),float(-l2*t12),0.0,0.0],[0.0,0.0,float(-l1*t14-l2*t20),float(l1*t17-l2*t22)],[0.0,0.0,float(-l2*t20),-float(l2*t22)]]).reshape(4,4)

    return np.transpose(Jee)

def dJG(q1, q2, q3, q4, dq1, dq2, dq3, dq4, l1, l2, w1, w2, ro):

    t3 = np.pi*(1.0/2.0);
    t4 = q1+t3;
    t2 = np.sin(t4);
    t5 = np.cos(t4);
    t6 = np.sin(q2);
    t7 = np.cos(q2);
    t8 = (w1);
    t9 = t2**2;
    t10 = l1*t6*t9;
    t11 = t5**2;
    t12 = l1*t6*t11;
    t13 = t10+t12;
    t14 = 1.0/t13;
    t15 = np.sin(t8);
    t16 = np.cos(t8);
    t17 = 1.0/t13**2;
    t18 = dq2*l1*t7*t11;
    t19 = dq2*l1*t7*t9;
    t20 = t18+t19;
    t21 = dq1*t2*t6;
    t22 = dq2*t2*t6;
    t33 = dq1*t5*t7;
    t34 = dq2*t5*t7;
    t23 = t21+t22-t33-t34;
    t24 = dq1*t2*t7;
    t25 = dq1*t5*t6;
    t26 = dq2*t2*t7;
    t27 = dq2*t5*t6;
    t28 = t24+t25+t26+t27;
    t29 = t2*t6;
    t30 = t2*t7;
    t31 = t5*t6;
    t32 = t30+t31;
    t35 = (ro);
    t36 = t29-t5*t7;
    t37 = l1*l2*t6*t11;
    t38 = l1*l2*t6*t9;
    t39 = t37+t38;
    t40 = 1.0/t39;
    t41 = 1.0/t39**2;
    t42 = dq2*l1*l2*t7*t11;
    t43 = dq2*l1*l2*t7*t9;
    t44 = t42+t43;
    t45 = dq1*l1*t5;
    t46 = dq1*l2*t5*t7;
    t47 = dq2*l2*t5*t7;
    t62 = dq1*l2*t2*t6;
    t63 = dq2*l2*t2*t6;
    t48 = t45+t46+t47-t62-t63;
    t49 = dq1*l1*t2;
    t50 = dq1*l2*t2*t7;
    t51 = dq1*l2*t5*t6;
    t52 = dq2*l2*t2*t7;
    t53 = dq2*l2*t5*t6;
    t54 = t49+t50+t51+t52+t53;
    t55 = l1*t2;
    t56 = l2*t2*t7;
    t57 = l2*t5*t6;
    t58 = t55+t56+t57;
    t59 = l1*t5;
    t60 = l2*t5*t7;
    t64 = l2*t2*t6;
    t61 = t59+t60-t64;
    t66 = q3+t3;
    t65 = np.sin(t66);
    t67 = np.cos(t66);
    t68 = np.sin(q4);
    t69 = np.cos(q4);
    t70 = (w2);
    t71 = t65**2;
    t72 = l1*t68*t71;
    t73 = t67**2;
    t74 = l1*t68*t73;
    t75 = t72+t74;
    t76 = 1.0/t75;
    t77 = np.sin(t70);
    t78 = np.cos(t70);
    t79 = 1.0/t75**2;
    t80 = dq4*l1*t69*t73;
    t81 = dq4*l1*t69*t71;
    t82 = t80+t81;
    t83 = dq3*t65*t68;
    t84 = dq4*t65*t68;
    t95 = dq3*t67*t69;
    t96 = dq4*t67*t69;
    t85 = t83+t84-t95-t96;
    t86 = dq3*t65*t69;
    t87 = dq3*t67*t68;
    t88 = dq4*t65*t69;
    t89 = dq4*t67*t68;
    t90 = t86+t87+t88+t89;
    t91 = t65*t68;
    t92 = t65*t69;
    t93 = t67*t68;
    t94 = t92+t93;
    t97 = l1*l2*t68*t73;
    t98 = l1*l2*t68*t71;
    t99 = t97+t98;
    t100 = 1.0/t99;
    t101 = 1.0/t99**2;
    t102 = dq4*l1*l2*t69*t73;
    t103 = dq4*l1*l2*t69*t71;
    t104 = t102+t103;
    t105 = dq3*l1*t67;
    t106 = dq3*l2*t67*t69;
    t107 = dq4*l2*t67*t69;
    t122 = dq3*l2*t65*t68;
    t123 = dq4*l2*t65*t68;
    t108 = t105+t106+t107-t122-t123;
    t109 = dq3*l1*t65;
    t110 = dq3*l2*t65*t69;
    t111 = dq3*l2*t67*t68;
    t112 = dq4*l2*t65*t69;
    t113 = dq4*l2*t67*t68;
    t114 = t109+t110+t111+t112+t113;
    t115 = l1*t65;
    t116 = l2*t65*t69;
    t117 = l2*t67*t68;
    t118 = t115+t116+t117;
    t119 = l1*t67;
    t120 = l2*t67*t69;
    t124 = l2*t65*t68;
    t121 = t119+t120-t124;
    dJG = np.array([[t14*t15*t23-t14*t16*t28+t15*t17*t20*t32+t16*t17*t20*t36,t15*t40*t48+t16*t40*t54-t15*t41*t44*t58+t16*t41*t44*t61,t76*t77*t85-t76*t78*t90+t77*t79*t82*t94+t78*t79*t82*(t91-t67*t69),t77*t100*t108+t78*t100*t114-t77*t101*t104*t118+t78*t101*t104*t121],[-t14*t16*t23-t14*t15*t28-t16*t17*t20*t32+t15*t17*t20*t36,-t16*t40*t48+t15*t40*t54+t16*t41*t44*t58+t15*t41*t44*t61,-t76*t78*t85-t76*t77*t90-t78*t79*t82*t94+t77*t79*t82*(t91-t67*t69),-t78*t100*t108+t77*t100*t114+t78*t101*t104*t118+t77*t101*t104*t121],[t14*t15*t28*t35+t14*t16*t35*(t21+t22-t33-t34)+t16*t17*t20*t32*t35-t15*t17*t20*t35*t36,t16*t35*t40*t48-t15*t35*t40*t54-t16*t35*t41*t44*t58-t15*t35*t41*t44*t61,-t35*t76*t78*t85-t35*t76*t77*t90+t35*t77*t79*t82*(t91-t67*t69)-t35*t78*t79*t82*t94,-t35*t78*t100*t108+t35*t77*t100*t114+t35*t78*t101*t104*t118+t35*t77*t101*t104*t121]]).reshape(3,4)

    return np.transpose(dJG)

def Gmap(w1, w2, ro):

    t2 = np.sin(w1);
    t3 = np.cos(w1);
    t4 = np.sin(w2);
    t5 = np.cos(w2);
    Gmap = np.array([[t3,t2,-ro*t2],[-t2,t3,-ro*t3],[t5,t4,ro*t4],[-t4,t5,ro*t5]]).reshape(4,3)

    return np.transpose(Gmap)