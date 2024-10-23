# geometric
Some geometric calculations

# Representation
## 2D
* Line
  * `[a, b, c]` for $$ax+by+c=0$$, where $$\sqrt{a^2+b^2}=1$$

* Circle
  * `[[a, b], r, [0.0, 0.0, 1.0, 0.0]]` for $$(x-a)^2+(y-b)^2=r^2$$
  * The last term, `[0.0, 0.0, 1.0, 0.0]]`, means that the circle lies in the plane $$z=0$$, which is ignorable

## 3D
* Line
  * `[[a, b, c], [r, s, t]]` for 
$$x=a+rk$$
$$y=b+sk$$
$$z=c+tk$$, where  $$\sqrt{r^2+s^2+t^2}=1,k\in\mathbb{R}$$
* Plane
  * `[a, b, c, d]` for $$ax+by+cz+d=0$$, where $$\sqrt{a^2+b^2+c^2}=1$$
* Circle
  * `[[x_c, y_c, z_c], r, [a, b, c, d]]` for circle lies in plane `[a, b, c, d]`, center at `[x_c, y_c, z_c]` and with radius `r`
* Sphere
  * `[[a, b, c], r]` for sphere center at `[a, b, c]` and with radius `r`