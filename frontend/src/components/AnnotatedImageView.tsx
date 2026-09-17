interface AnnotatedImageViewProps {
  base64Png: string;
  width: number;
  height: number;
}

export function AnnotatedImageView({ base64Png, width, height }: AnnotatedImageViewProps) {
  return (
    <div className="annotated-image">
      <img
        src={`data:image/png;base64,${base64Png}`}
        alt="Annotated shelf with detected facings, region bands, and stock gaps"
        width={width}
        height={height}
      />
    </div>
  );
}
